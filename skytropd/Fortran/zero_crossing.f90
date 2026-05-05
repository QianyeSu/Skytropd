module zero_crossing_mod
  use, intrinsic :: iso_c_binding
  use, intrinsic :: ieee_arithmetic
contains
  real(c_double) function sign_no_nan(x)
    real(c_double), intent(in) :: x

    if (x > 0.0d0) then
      sign_no_nan = 1.0d0
    else if (x < 0.0d0) then
      sign_no_nan = -1.0d0
    else
      sign_no_nan = 0.0d0
    end if
  end function sign_no_nan

  subroutine zc_fortran(f, nbands, nlat, lat, lat_uncertainty, zc) bind(C, name="zc_fortran")
    integer(c_int), value :: nbands, nlat
    real(c_double), intent(in) :: f(*)
    real(c_double), intent(in) :: lat(*)
    real(c_double), value :: lat_uncertainty
    real(c_double), intent(out) :: zc(*)

    integer :: i, j, base, count, a0, a2
    integer :: di, first_di
    real(c_double) :: x, y, nan_value
    logical :: any_pos, any_neg

    nan_value = ieee_value(0.0d0, ieee_quiet_nan)

    do i = 0, nbands - 1
      zc(i + 1) = nan_value
      any_pos = .false.
      any_neg = .false.
      base = i * nlat

      do j = 1, nlat
        x = f(base + j)
        if (x > 0.0d0) any_pos = .true.
        if (x < 0.0d0) any_neg = .true.
      end do
      if (.not. (any_pos .and. any_neg)) cycle

      count = 0
      a0 = -1
      a2 = -1
      first_di = 0

      do j = 1, nlat - 1
        x = f(base + j)
        y = f(base + j + 1)
        if (ieee_is_nan(x) .or. ieee_is_nan(y)) cycle
        di = int(sign_no_nan(y) - sign_no_nan(x))
        if (di /= 0) then
          count = count + 1
          if (count == 1) then
            a0 = j
            first_di = di
          else if (count == 3) then
            a2 = j
          end if
        end if
      end do

      if (count == 0) cycle
      if (count > 2 .and. abs(lat(a2) - lat(a0)) < lat_uncertainty) cycle

      if (abs(first_di) == 1) then
        zc(i + 1) = lat(a0 + 1)
      else
        x = f(base + a0)
        y = f(base + a0 + 1)
        zc(i + 1) = -x * (lat(a0 + 1) - lat(a0)) / (y - x) + lat(a0)
      end if
    end do
  end subroutine zc_fortran

  subroutine descending_threshold_fortran( &
      profile, nbands, nlat, lat, start_idx, threshold, out &
  ) bind(C, name="descending_threshold_fortran")
    integer(c_int), value :: nbands, nlat
    real(c_double), intent(in) :: profile(*)
    real(c_double), intent(in) :: lat(*)
    integer(c_int), intent(in) :: start_idx(*)
    real(c_double), intent(in) :: threshold(*)
    real(c_double), intent(out) :: out(*)

    integer :: i, j, base, start1
    real(c_double) :: val0, val1, nan_value

    nan_value = ieee_value(0.0d0, ieee_quiet_nan)

    do i = 0, nbands - 1
      out(i + 1) = nan_value
      base = i * nlat
      start1 = start_idx(i + 1) + 1

      if (start1 < 1 .or. start1 >= nlat) cycle

      do j = start1, nlat - 1
        val0 = profile(base + j)
        val1 = profile(base + j + 1)
        if (ieee_is_nan(val0) .or. ieee_is_nan(val1)) cycle

        if (val0 >= threshold(i + 1) .and. val1 <= threshold(i + 1)) then
          if (abs(val0 - val1) <= epsilon(val0)) then
            out(i + 1) = 0.5d0 * (lat(j) + lat(j + 1))
          else
            out(i + 1) = lat(j) &
                + (threshold(i + 1) - val0) * (lat(j + 1) - lat(j)) &
                / (val1 - val0)
          end if
          exit
        end if
      end do
    end do
  end subroutine descending_threshold_fortran
end module zero_crossing_mod
