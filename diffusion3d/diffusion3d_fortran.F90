subroutine diffusion_fortran_step(p0, p1, nx, ny, nz, &
     ce, cw, cn, cs, ct, cb, cc)
  implicit none
  integer, intent(in) :: nx, ny, nz
  real, dimension(nx, ny, nz), intent(in) :: p0
  real, dimension(nx, ny, nz), intent(out) :: p1
  real, intent(in) :: ce, cw, cn, cs, ct, cb, cc
  real :: c, w, e, n, s, t, b
  integer :: i, j, k

  !$acc parallel loop present(p0, p1)
  do k = 1, nz
     do j = 1, ny
        do i = 1, nx
           c = p0(i, j, k)
           if (i == 1) then
              w = c
           else
              w = p0(i-1, j, k)
           end if
           if (i == nx) then
              e = c
           else
              e = p0(i+1, j, k)
           end if
           if (j == 1) then
              s = c
           else
              s = p0(i, j-1, k)
           end if
           if (j == ny) then
              n = c
           else
              n = p0(i, j+1, k)
           end if
           if (k == 1) then
              b = c
           else
              b = p0(i, j, k-1)
           end if
           if (k == nz) then
              t = c
           else
              t = p0(i, j, k+1)
           end if
           p1(i, j, k) = cc * c + cw * w + ce * e + &
                cs * s + cn * n + cb * b + ct * t
        end do
     end do
  end do
end subroutine diffusion_fortran_step

subroutine diffusion_fortran(p0, p1, nx, ny, nz, &
     ce, cw, cn, cs, ct, cb, cc, count)
  implicit none
  integer, intent(in) :: nx, ny, nz
  real, dimension(nx, ny, nz), intent(inout) :: p0, p1
  real, intent(in) :: ce, cw, cn, cs, ct, cb, cc
  integer, intent(in) :: count
  integer :: l
  
!$acc data copy(p0, p1)
  do l = 1, count / 2
     call diffusion_fortran_step(p0, p1, nx, ny, nz, &
          ce, cw, cn, cs, ct, cb, cc)
     call diffusion_fortran_step(p1, p0, nx, ny, nz, &
          ce, cw, cn, cs, ct, cb, cc)
  end do
!$acc end data
  
end subroutine diffusion_fortran
