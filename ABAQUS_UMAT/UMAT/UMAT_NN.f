C ========================================================================== C
C UMAT-NN: written by Chulmin Kweon & Hyoung Suk Suh (Columbia Univ.)        C
C >> Note: file location must be specified in Lines 682,686,752,756,760,764  C
C >>       they should contain the "absolute" path.                          C
C ========================================================================== C     
      subroutine convert_123_to_prt(sigma1, sigma2, sigma3, pp, rho, theta)
        ! convert (sigma1,sigma2,sigma3) to (p, rho, theta)
        real*8 sigma1, sigma2, sigma3
        real*8 pp, rho, theta
        real*8 sigma1_pp, sigma2_pp, sigma3_pp
        
        sigma1_pp = sqrt(2.d0)/2.d0 * sigma1 - sqrt(2.d0)/2.d0 * sigma3
        sigma2_pp = -sqrt(6.d0)/6.d0 * sigma1
     $              +sqrt(6.d0)/3.d0 * sigma2
     $              -sqrt(6.d0)/6.d0 * sigma3
        sigma3_pp =  sqrt(3.d0)/3.d0 * sigma1
     $              +sqrt(3.d0)/3.d0 * sigma2
     $              +sqrt(3.d0)/3.d0 * sigma3
                    
        rho   = sqrt(sigma1_pp**2 + sigma2_pp**2)
        theta = atan2(sigma2_pp, sigma1_pp)
        pp    = 1.d0/sqrt(3.d0) * sigma3_pp
        
        if (theta .lt. 0.d0) then
            theta = theta + 2.d0 * 3.141592653589793
        end if
      end subroutine convert_123_to_prt


      subroutine Jacobi(inpMat, eigvl, eigvc, abserr)
        ! compute eigenvalues & eigenvectors

        implicit none
        integer i, j, k, n
        integer srtind(3), tmpint

        double precision inpMat(3,3)
        double precision eigvl(3)        
        double precision eigvc(3,3)
        double precision a(3,3), x(3,3)
        double precision abserr, b2, bar, tmp
        double precision beta, coeff, c, s, cs, sc

        n = 3
        
        do i =1,n
          do j =1,n
            a(i,j) = inpMat(i,j)
          end do
        end do

        x = 0.0
        do i=1,n
          x(i,i) = 1.0
        end do

        b2 = 0.0
        do i=1,n
          do j=1,n
            if (i.ne.j) b2 = b2 + a(i,j)**2
          end do
        end do
        
        if (b2 .gt. abserr) then

          bar = 0.5*b2/float(n*n)
          
          do while (b2.gt.abserr)
            do i=1,n-1
              do j=i+1,n
                if (a(j,i)**2 <= abs(bar)) cycle  
                b2 = b2 - 2.0*a(j,i)**2
                bar = 0.5*b2/float(n*n)
                beta = (a(j,j)-a(i,i))/(2.0*a(j,i))
                coeff = 0.5*beta/sqrt(1.0+beta**2)
                s = sqrt(max(0.5+coeff,0.0))
                c = sqrt(max(0.5-coeff,0.0))
                do k=1,n
                  cs =  c*a(i,k)+s*a(j,k)
                  sc = -s*a(i,k)+c*a(j,k)
                  a(i,k) = cs
                  a(j,k) = sc
                end do
                do k=1,n
                  cs =  c*a(k,i)+s*a(k,j)
                  sc = -s*a(k,i)+c*a(k,j)
                  a(k,i) = cs
                  a(k,j) = sc
                  cs =  c*x(k,i)+s*x(k,j)
                  sc = -s*x(k,i)+c*x(k,j)
                  x(k,i) = cs
                  x(k,j) = sc
                end do
              end do
            end do
          end do
        end if

        do i =1,n
          do j = 1,n
            eigvc(i,j) = x(i,j)
          end do
        end do

        eigvl(1) = a(1,1)
        eigvl(2) = a(2,2)
        eigvl(3) = a(3,3)

      end subroutine Jacobi


      subroutine matrixinv(a,b,n)
        ! calculate the inverse of a matrix
        integer :: i,j,k,l,m,n,irow
        real(8) :: big,a(n,n),b(n,n),dum

        do i = 1,n
          do j = 1,n
            b(i,j) = 0.0
          enddo
          b(i,i) = 1.0
        enddo

        do i = 1,n
          big = a(i,i)
          do j = i,n
            if (a(j,i).gt.big) then
              big = a(j,i)
              irow = j
            endif
          enddo

          if (big.gt.a(i,i)) then
            do k = 1,n
              dum = a(i,k) 
              a(i,k) = a(irow,k)
              a(irow,k) = dum
              dum = b(i,k) 
              b(i,k) = b(irow,k)
              b(irow,k) = dum
            enddo
          endif

          dum = a(i,i)
          do j = 1,n
            a(i,j) = a(i,j)/dum
            b(i,j) = b(i,j)/dum
          enddo
        
          do j = i+1,n
            dum = a(j,i)
            do k = 1,n
              a(j,k) = a(j,k) - dum*a(i,k)
              b(j,k) = b(j,k) - dum*b(i,k)
            enddo
          enddo
        enddo

        do i = 1,n-1
          do j = i+1,n
            dum = a(i,j)
            do l = 1,n
              a(i,l) = a(i,l)-dum*a(j,l)
              b(i,l) = b(i,l)-dum*b(j,l)
            enddo
          enddo
        enddo
      end subroutine matrixinv


      subroutine TensorTovoigt(Ce, ddsdde)
        ! convert tensor notation to Voigt notation
        real*8 Ce(3,3,3,3), ddsdde(6,6)
        integer i,j,k,l
        do i = 1,3
          do j = 1,3
            ddsdde(i,j) = Ce(i,i,j,j)
          end do
        end do
        
        do i = 1,3
          do j = 4,6
            k = mod(-j+7,3)+1
            l = mod(-j+8,3)+1
            ddsdde(i,j) = Ce(i,i,k,l)
          end do
        end do
        
        do i = 4,6
          do j = 1,3
            k = mod(-i+7,3)+1
            l = mod(-i+8,3)+1
            ddsdde(i,j) = Ce(k,l,j,j)
          end do
        end do
        
        do i = 4,6
          do j = 4,6
            k1 = mod(-i+7,3)+1
            l1 = mod(-i+8,3)+1
            k2 = mod(-j+7,3)+1
            l2 = mod(-j+8,3)+1
            ddsdde(i,j) = Ce(k1,l1,k2,l2)
          end do
        end do
      end subroutine TensorTovoigt


      subroutine identity_4(EYE4)
        ! fourth order identity tensor
        real*8 EYE2(3,3), EYE4(3,3,3,3) 
        integer nDim
        nDim = 3
        EYE2 = 0.d0
        EYE2(1,1) = 1.d0
        EYE2(2,2) = 1.d0
        EYE2(3,3) = 1.d0
        
        do i = 1, nDim
          do j = 1, nDim
            do k = 1, nDim
              do l = 1, nDim
                EYE4(i,j,k,l) = (EYE2(i,l) * EYE2(j,k)
     $                         + EYE2(i,k) * EYE2(j,l)) / 2.d0
              end do 
            end do
          end do
        end do
  
      end subroutine identity_4


      subroutine tensor_oMult(A, B, res)
        ! tensor product
        real*8 A(3,3), B(3,3), res(3,3,3,3) 
        integer nDim

        nDim = 3
        do i = 1,nDim
          do j = 1,nDim
            do k = 1,nDim
              do l = 1,nDim
                res(i,j,k,l) = A(i,j) * B(k,l)
              end do
            end do
          end do
        end do
      end subroutine tensor_oMult


      subroutine gauss_2(a,b,x,n)
        ! solve system of equations: Ax = b

        implicit none
        integer n
        real*8 a(n,n), b(n), x(n)
        real*8 s(n)
        real*8 c, pivot, store
        integer i, j, k, l

        do k=1, n-1
          do i=k,n
            s(i) = 0.0
            do j=k,n
            s(i) = max(s(i),abs(a(i,j)))
            end do
          end do

          pivot = abs(a(k,k)/s(k))
          l=k
          do j=k+1,n
            if(abs(a(j,k)/s(j)) > pivot) then
              pivot = abs(a(j,k)/s(j))
              l=j
            end if
          end do

          if(pivot == 0.0) then
            write(*,*) " The matrix is sigular "
          return
          end if

          if (l /= k) then
            do j=k,n
              store = a(k,j)
              a(k,j) = a(l,j)
              a(l,j) = store
            end do
            store = b(k)
            b(k) = b(l)
            b(l) = store
          end if

          do i=k+1,n
            c=a(i,k)/a(k,k)
            a(i,k) = 0.0
            b(i)=b(i)- c*b(k)
            do j=k+1,n
              a(i,j) = a(i,j)-c*a(k,j)
            end do
          end do
        end do

        x(n) = b(n)/a(n,n)
        do i=n-1,1,-1
          c=0.0
          do j=i+1,n
            c= c + a(i,j)*x(j)
          end do
          x(i) = (b(i)- c)/a(i,i)
        end do
      end subroutine gauss_2


      subroutine NN_scaleback(scaled, scaler, Nout, output)
        ! Inverse scaling for NN function

        real*8 scaled(7 ) 
        real*8 output(7 )
        real*8 scaler(14)
        
        output = 0.d0
        do i = 1, Nout
          output(i) = scaled(i) * (scaler(2*i) - scaler(2*i-1)) + scaler(2*i-1)
        end do
      end subroutine NN_scaleback


      subroutine NN_scale(xin, scaler, Nin, scaled)
        ! Scaling for NN function
        real*8 xin(7 )
        real*8 scaled(7 )
        real*8 scaler(14)

        scaled = 0.d0
        do i = 1, Nin
          scaled(i) = (xin(i) - scaler(2*i-1)) /(scaler(2*i) - scaler(2*i-1))
        end do 
      end subroutine NN_scale


      subroutine ReadScaler(fname, Ninput, scaler)
        ! Read scaler for NN function
        character*100 fname
        integer eof, fid
        integer Ninput
        real*8  scaler(14)
        real*8  tmpreal
        
        scaler = 0.d0
        OPEN(unit = fid, file=fname, status='old')
        do i = 1, Ninput*2
          read(fid, fmt=*, IOSTAT=eof) tmpreal
          scaler(i) = tmpreal
        end do
        CLOSE (fid, STATUS='KEEP')
      end subroutine ReadScaler


      subroutine ReadLayer(fname, Nlayer, NNodes, Layers, ActFun)
        ! Read NN layers
        integer Nlayer, NNodes(10), Nlines

        integer eof, fid, fid1
        character*10 reader
        integer reintr
        integer tmp
              
        character*100 fname
        character*10, dimension(10) :: Layers
        character*10, dimension(10) :: ActFun
        
        Nlayer = 0
        Nlines = 0

        open(unit = fid, file=fname, status='old')
        read(fid, fmt=*, IOSTAT=eof) Nlines
        read(fid, fmt=*, IOSTAT=eof) reader, reintr
        
        if (reader .ne. "input") then
          print *, "Check the NN_layer file"
          read(*,*)
        end if
        i =1
        NNodes(i) = reintr
        i = i +1
        j = 1

        do n = 2, Nlines
          read(fid, fmt=*, IOSTAT=eof) reader, reintr
          if( mod(n,2) .eq. 0) then
            NNodes(i) = reintr
            Layers(j) = reader
            i = i + 1
          else
            ActFun(j) = reader
            j=j+1
          end if
        end do
        Nlayer = (Nlines + 1) / 2 

        close(fid, STATUS='KEEP')
      end subroutine ReadLayer


      subroutine DerivRelu(xin, xout)
        ! Derivative of ReLU activation function
        real*8 xin, xout
          if (xin .le. 0.d0) then
            xout = 0.d0
          else
            xout = 1.d0
          end if
      end subroutine DerivRelu


      subroutine relu(xin, xout)
        ! ReLU activation function
        real*8 xin, xout
          if (xin .lt. 0.d0) then
            xout = 0.d0
          else
            xout = xin
          end if
      end subroutine relu

      
      subroutine DerivSigmoid(xin, xout)
        ! Derivative of Sigmoid activation function
        real*8 xin, xout
          xout = 1.d0/(1.d0 + exp(-xin)) * (1.d0 - 1.d0/(1.d0 + exp(-xin)))
      end subroutine DerivSigmoid


      subroutine Sigmoid(xin, xout)
        ! Sigmoid activation function
        real*8 xin, xout
          xout = 1.d0/(1.d0 + exp(-xin))
      end subroutine Sigmoid


      subroutine DerivTanh(xin, xout)
        ! Derivative of tanh activation function
        real*8 xin, xout
          xout = 1.d0 - tanh(xin)**2
      end subroutine DerivTanh


      subroutine NN_output(xin, Nlayer, NNodes, layers, ActFun, 
     $   inscaler, outscaler, WArray, BArray, ff)
        ! NN function
        integer Nlayer, NNodes(10)
        real*8 xin(7 ), xinscald(7 ), xin_prt(7 )
        real*8  ff(7 ), foscld(7 )
        real*8 WArray(150,150,10), BArray(150,10)
        real*8 xoLayr(150), xiLayr(150), reLayr(150)
        real*8 pp, rho, theta
        integer tmp
        
        character*10, dimension(10) :: Layers
        character*10, dimension(10) :: ActFun

        real*8  inscaler(14)
        real*8 outscaler(14)

        call convert_123_to_prt(xin(1), xin(2), xin(3), pp, rho, theta)
        xin_prt = xin 
        xin_prt(1) = pp
        xin_prt(2) = rho
        xin_prt(3) = theta

        call NN_scale(xin_prt, inscaler, NNodes(1), xinscald)
 
        xoLayr = 0.d0
        reLayr = 0.d0

        do i = 1,NNodes(1)
          xiLayr(i) = xinscald(i)
        end do

        do i = 1,Nlayer-1         
          if (Layers(i) .eq. "dense") then
            tmp = NNodes(i)
            j = i
            do while (tmp .eq. 0 ) 
              j = j -1
              tmp = NNodes(j)
            end do
            
            do j = 1, NNodes(i+1)
              xoLayr(j) = 0.d0
              do k = 1, tmp
                xoLayr(j)  = xoLayr(j) + WArray(j,k,i) * xiLayr(k)
              end do
              xoLayr(j) = xoLayr(j) + BArray(j,i)
            end do
            
          else if (Layers(i) .eq. "multiply") then
            tmp = NNodes(i)
            j = i 
            do while (tmp .eq. 0 ) 
              j = j -1
              tmp = NNodes(j)
            end do

            do j = 1, tmp
              xoLayr(j) = xiLayr(j) * xiLayr(j)
            end do
          end if 
          
          if (i .ne. Nlayer) then
            tmp = NNodes(i+1)
            j = i 
            do while (tmp .eq. 0 ) 
              tmp = NNodes(j)
              j = j -1
            end do
            if (ActFun(i) .eq. "relu") then 
              do j = 1, tmp
                call relu(xoLayr(j), reLayr(j))     
                xoLayr(j) = reLayr(j)
              end do

            else if (ActFun(i) .eq. "sigmoid") then 
              do j = 1, tmp
                call Sigmoid(xoLayr(j), reLayr(j))     
                xoLayr(j) = reLayr(j)
              end do

            else if (ActFun(i) .eq. "tanh") then 
              do j = 1, tmp
                reLayr(j) = tanh(xoLayr(j))
                xoLayr(j) = reLayr(j)
              end do
            end if
          end if
          
          tmp = NNodes(i+1)
          j = i
          do while (tmp .eq. 0 ) 
            tmp = NNodes(j)
            j = j-1
          end do

          do j = 1, tmp
            xiLayr(j) =  xoLayr(j)
          end do
        end do
        
        do i = 1, NNodes(Nlayer)
          foscld(i) = xiLayr(i)
        end do

        call NN_scaleback(foscld, outscaler, NNodes(Nlayer), ff)
      end subroutine NN_output





C =================================================================== C
      subroutine umat(stress,statev,ddsdde,sse,spd,scd,
     $   rpl,ddsddt,drplde,drpldt,
     $   stran,dstran,time,dtime,temp,dtemp,predef,dpred,cmname,
     $   ndi,nshr,ntens,nstatv,props,nprops,coords,drot,pnewdt,
     $   celent,dfgrd0,dfgrd1,noel,npt,layer,kspt,jstep,kinc)
        ! UMAT subroutine
     
        include 'ABA_PARAM.INC'

        character*80 cmname
        dimension stress(6),statev(13),
     $    ddsdde(6,6),ddsddt(6),drplde(6),
     $    stran(6),dstran(6),time(2),predef(1),dpred(1),
     $    props(2),coords(3),drot(3,3),dfgrd0(3,3),dfgrd1(3,3),
     $    jstep(4)

        integer i,j,k,l

        ! Neural network input and output
        real*8 xin(7), xindist(7)
        real*8 fout(7), ff(7)
        real*8 df(7)
        real*8 d2f(7,7)
        real*8 dfdist(7)

        ! finite difference for 2nd order derivatives
        real*8 dfdsig1dist1(3), dfdsig2dist2(3)
        real*8 dfdsig3dist3(3), dfdlamdadist4(3)

        ! Define iscalr, oscalr
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! The maximum number of inputscaler and outputscaler
        ! are fixed as "14" here for convenience.
        ! Make sure to change the variable size in functions
        ! including "NN_scaleback", "NN_scale", "NN_output"
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        real*8 iscalr_f(14), oscalr_f(14)
        real*8 iscalr_df(14), oscalr_df(14)

        ! Define tensors
        real*8 EYE2(3,3), EYE4(3,3,3,3)
        real*8 xIXI(3,3,3,3)
        real*8 xm1(3,3), xm2(3,3), xm3(3,3)
        
        ! Define tangent operator
        real*8 Ce(3,3,3,3)
        real*8 DD(6,6), HH(6,6)
        real*8 d2fdsig2(3,3,3,3)
        real*8 d2fdsig2_voigt(6,6)
        real*8 tmp(6,6)
        real*8 tmpinv(6,6)
        real*8 dfdsig(3,3)
        real*8 dfdsig_voigt(6)

        ! Define strain tensors & strain invariants
        real*8 eps(3,3), epse(3,3), epsp(3,3), deps(3,3)
        real*8 epse_tr(3,3), eps_n1(3,3)
        real*8 epse_cur(3)

        ! Define eigenvalues & eigenvectors
        real*8 epse_eig_tr(3)
        real*8 eps_eig_n1(3)
        real*8 epse_eigvc_tr(3,3)
        real*8 epse_eig(3)
        real*8 eigvctmp(3,3)

        ! Define plastic multiplier
        real*8 xlamda, dlamda

        ! Newton-Raphson iteration      
        ! Define stress tensors
        real*8 sig(3,3), sig_tr(3,3)
        real*8 sig_eig_tr(3), sig_eig_cur(3)
        
        ! Define residual, target variable vectors & jacobian
        real*8 res(4), xx(4), dxx(4)
        real*8 xjac(4,4)      
        
        real*8 props1(4)

        ! Define Newton-Raphson parameters
        parameter(toler=1.d-8, newton=40)
        
        ! NN parameters 
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! The maximum number of layer is fixed as "10" here
        ! for convenience.
        ! Make sure to change the variable size in functions
        ! including "NN_output", "ReadLayer"
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        integer iread

        real*8 WArr_f(150,150,10)
        real*8 BArr_f(150,10)

        integer Nlayer_f, NNode_f(10)
        character*10, dimension(10) :: Layers_f
        character*10, dimension(10) :: ActFun_f

        real*8 WArr_df(150,150,10)
        real*8 BArr_df(150,10)

        integer Nlayer_df, NNode_df(10)
        character*10, dimension(10) :: Layers_df
        character*10, dimension(10) :: ActFun_df

        character*100 fname

        integer tmpint
        
        real*8 C_algo(3,3,3,3), aAB(3,3), xjac_tmp(3,3), xjac_tmpinv(3,3)
        real*8 xm11(3,3), xm12(3,3), xm13(3,3)  
        real*8 xm21(3,3), xm22(3,3), xm23(3,3)  
        real*8 xm31(3,3), xm32(3,3), xm33(3,3)  


        ! =================== STRESS INTEGRATION STARTS =================== !
        ! Read the NN layer information from "NN_params_*.txt"
        NNode_f = 0
        fname = "C:\temp\\ABAQUS_UMAT\Tresca\NN_input\NN_params_f.txt"
        call ReadLayer(fname, Nlayer_f, NNode_f, Layers_f, ActFun_f)

        NNode_df =0
        fname = "C:\temp\ABAQUS_UMAT\Tresca\NN_input\NN_params_df.txt"
        call ReadLayer(fname, Nlayer_df, NNode_df, Layers_df, ActFun_df)
        
        ! Read the NN parameters from the ".inp" file
        i = 1
        iread = 3
        do while(i .LT. Nlayer_f)
          do j = 1,NNode_f(i+1)
            BArr_f(j,i) = props(iread)
            iread = iread + 1
          end do
          i = i + 1
        end do

        i = 1
        do while(i .LT. Nlayer_f)
          j = i 
          tmpint = NNode_f(j)
          do while (tmpint .eq. 0) 
            j = j-1
            tmpint = NNode_f(j) 
          end do
          
          do j = 1,NNode_f(i+1) * tmpint
            k = (j-1)/tmpint+1
            l = mod(j,tmpint)
            if (l .eq. 0) then
              l = tmpint
            end if
            WArr_f(k,l,i) = props(iread)
            iread = iread + 1
          end do
          i = i + 1
        end do

        i = 1
        do while(i .LT. Nlayer_df)
          do j = 1,NNode_df(i+1)
            BArr_df(j,i) = props(iread)
            iread = iread + 1
          end do
          i = i + 1
        end do

        i = 1
        do while(i .LT. Nlayer_df)
          j = i 
          tmpint = NNode_df(j)
          do while (tmpint .eq. 0) 
            j = j-1
            tmpint = NNode_f(j) 
          end do
          
          do j = 1,NNode_df(i+1) * tmpint
            k = (j-1)/tmpint+1
            l = mod(j,tmpint)
            if (l .eq. 0) then
              l = tmpint
            end if
            WArr_df(k,l,i) = props(iread)
            iread = iread + 1
          end do
          i = i + 1
        end do

        ! Load outputscaler for f
        fname = "C:\temp\ABAQUS_UMAT\Tresca\NN_input\fouts.txt"
        call ReadScaler(fname, NNode_f(Nlayer_f), oscalr_f)

        ! Load outputscaler for df
        fname = "C:\temp\ABAQUS_UMAT\Tresca\NN_input\dfouts.txt"
        call ReadScaler(fname, NNode_df(Nlayer_df), oscalr_df)

        ! Load inputscaler for f    
        fname = "C:\temp\ABAQUS_UMAT\Tresca\NN_input\fins.txt"
        call ReadScaler(fname, NNode_f(1), iscalr_f)
        
        ! Load inputscaler for df    
        fname = "C:\temp\ABAQUS_UMAT\Tresca\NN_input\dfins.txt"
        call ReadScaler(fname, NNode_df(1), iscalr_df)

        ! Define unit 2nd order tensor
        EYE2 = 0.d0
        EYE2(1,1) = 1.d0
        EYE2(2,2) = 1.d0
        EYE2(3,3) = 1.d0

        call tensor_oMult(EYE2, EYE2, xIXI)
        call identity_4(EYE4)

        ! Read properties
        E   = props(1)
        xnu = props(2)

        ! Read stress from previous time step
        sig(1,1) = stress(1)
        sig(2,2) = stress(2)
        sig(3,3) = stress(3)
        sig(1,2) = stress(4)
        sig(1,3) = stress(5)
        sig(2,3) = stress(6)
        sig(2,1) = stress(4)
        sig(3,1) = stress(5)
        sig(3,2) = stress(6)

        ! Read strain & strain increment from previous time step
        eps(1,1) = stran(1)
        eps(2,2) = stran(2)
        eps(3,3) = stran(3)
        eps(1,2) = stran(4) / 2.d0
        eps(1,3) = stran(5) / 2.d0
        eps(2,3) = stran(6) / 2.d0
        eps(2,1) = stran(4) / 2.d0
        eps(3,1) = stran(5) / 2.d0
        eps(3,2) = stran(6) / 2.d0

        deps(1,1) = dstran(1)
        deps(2,2) = dstran(2)
        deps(3,3) = dstran(3)
        deps(1,2) = dstran(4) / 2.d0
        deps(1,3) = dstran(5) / 2.d0
        deps(2,3) = dstran(6) / 2.d0
        deps(2,1) = dstran(4) / 2.d0
        deps(3,1) = dstran(5) / 2.d0
        deps(3,2) = dstran(6) / 2.d0
        
        ! Read epse from previous time step
        epse(1,1) = statev(1)
        epse(2,2) = statev(2)
        epse(3,3) = statev(3)
        epse(1,2) = statev(4) / 2.d0
        epse(1,3) = statev(5) / 2.d0
        epse(2,3) = statev(6) / 2.d0
        epse(2,1) = statev(4) / 2.d0
        epse(3,1) = statev(5) / 2.d0
        epse(3,2) = statev(6) / 2.d0
        
        ! Read epsp from previous time step
        epsp(1,1) = statev(7)
        epsp(2,2) = statev(8)
        epsp(3,3) = statev(9)
        epsp(1,2) = statev(10) / 2.d0
        epsp(1,3) = statev(11) / 2.d0
        epsp(2,3) = statev(12) / 2.d0
        epsp(2,1) = statev(10) / 2.d0
        epsp(3,1) = statev(11) / 2.d0
        epsp(3,2) = statev(12) / 2.d0
        
        ! Read xlamda from previous time step
        xlamda = statev(13)
        
        ! Construct elasticity tensor (ddsdde)
        eG   =  E/(2.d0 * (1.d0+xnu))
        eLam = (E/(1.d0 -  2.d0*xnu)-2.d0*eG)/3.d0
        eK   =  E/3.d0 / (1.d0-2.d0*xnu)

        aa = eK + (4.d0/3.d0)*eG
        bb = eK - (2.d0/3.d0)*eG

        dsig1depse1 = aa
        dsig1depse2 = bb
        dsig1depse3 = bb
        dsig2depse1 = bb
        dsig2depse2 = aa
        dsig2depse3 = bb
        dsig3depse1 = bb
        dsig3depse2 = bb
        dsig3depse3 = aa

        ! LOADING step
        ! Compute trial strain
        epse_tr = epse + deps
        call Jacobi(epse_tr, epse_eig_tr, eigvctmp, toler)

        ! Compute tangent       
        do i = 1, 3
          do j = 1, 3
            do k = 1, 3
              do l = 1, 3
                Ce(i,j,k,l) = eLam * xIxI(i,j,k,l)
     $                + 2.d0 * eG * EYE4(i,j,k,l)
              end do 
            end do 
          end do
        end do
        
        ! Compute trial stress
        sig_tr = 0.d0      
        do i = 1, 3
          do j = 1, 3
            do k = 1, 3
              do l = 1, 3
                sig_tr(i,j) = sig_tr(i,j) + Ce(i,j,k,l) * epse_tr(k,l)
              end do 
            end do 
          end do
        end do

        call Jacobi(sig_tr, sig_eig_tr, eigvctmp, toler)
        
        do i = 1,3
          do j =1,3
            xm1(i,j) = eigvctmp(i,1) * eigvctmp(j,1)
            xm2(i,j) = eigvctmp(i,2) * eigvctmp(j,2)
            xm3(i,j) = eigvctmp(i,3) * eigvctmp(j,3)
          end do
        end do

        xin = 0.d0
        xin(1) = sig_eig_tr(1)
        xin(2) = sig_eig_tr(2)
        xin(3) = sig_eig_tr(3)
        xin(4) = xlamda

        call NN_output(xin, Nlayer_f, NNode_f, layers_f,
     $       ActFun_f, iscalr_f, oscalr_f, WArr_f, BArr_f, fout)
          
        if (fout(1) .le. 0.d0) then
          sig = sig_eig_tr(1) * xm1
     $        + sig_eig_tr(2) * xm2
     $        + sig_eig_tr(3) * xm3
          
          epse = epse_tr
          
          eps = epse + epsp
          
          call TensorTovoigt(Ce, ddsdde)

        else
          epse_eig = 0.d0
          eigvctmp = 0.d0
          call jacobi(epse, epse_eig, eigvctmp, toler)
          
          dlamda = 0.d0

          xx(1) = epse_eig(1)
          xx(2) = epse_eig(2)
          xx(3) = epse_eig(3)
          xx(4) = dlamda

          ! Newton-Raphson iteration
          kewton = 0
          err = 100.d0
          do while((err .gt. toler) .and. (kewton .lt. newton+1))
            kewton = kewton +1

            res = 0.d0
            xjac = 0.d0

            epse_cur(1) = xx(1) 
            epse_cur(2) = xx(2) 
            epse_cur(3) = xx(3) 
            xlamda_cur = xlamda + xx(4)

            sig_eig_cur(1) = aa * epse_cur(1) + bb * epse_cur(2) + bb * epse_cur(3)
            sig_eig_cur(2) = bb * epse_cur(1) + aa * epse_cur(2) + bb * epse_cur(3)
            sig_eig_cur(3) = bb * epse_cur(1) + bb * epse_cur(2) + aa * epse_cur(3)

            xin(1) = sig_eig_cur(1)
            xin(2) = sig_eig_cur(2)
            xin(3) = sig_eig_cur(3)
            xin(4) = xlamda_cur

            call NN_output(xin, Nlayer_df, NNode_df, Layers_df, 
     $      ActFun_df, iscalr_df, oscalr_df, WArr_df, BArr_df, df)

            dfdsig1 = df(1)
            dfdsig2 = df(2)
            dfdsig3 = df(3)          

            dist1 = 1d-7 * (iscalr_df(2) - iscalr_df(1))
            dist2 = 1d-7 * (iscalr_df(4) - iscalr_df(3))
            dist3 = 1d-7 * (iscalr_df(6) - iscalr_df(5))
            dist4 = 1d-7 * (iscalr_df(8) - iscalr_df(7))

            xindist = xin
            xindist(1) = xindist(1) + dist1

            call NN_output(xindist, Nlayer_df, NNode_df, Layers_df, 
     $      ActFun_df, iscalr_df, oscalr_df, WArr_df, BArr_df, dfdsig1dist1)

            xindist = xin
            xindist(2) = xindist(2) + dist2
            
            call NN_output(xindist, Nlayer_df, NNode_df, Layers_df, 
     $      ActFun_df, iscalr_df, oscalr_df, WArr_df, BArr_df, dfdsig2dist2)

            xindist = xin
            xindist(3) = xindist(3) + dist3

            call NN_output(xindist, Nlayer_df, NNode_df, Layers_df, 
     $      ActFun_df, iscalr_df, oscalr_df, WArr_df, BArr_df, dfdsig3dist3)

            d2fdsig1dsig1 = (dfdsig1dist1(1) - dfdsig1) / dist1
            d2fdsig2dsig2 = (dfdsig2dist2(2) - dfdsig2) / dist2
            d2fdsig3dsig3 = (dfdsig3dist3(3) - dfdsig3) / dist3

            d2fdsig1dsig2 = (dfdsig2dist2(1) - dfdsig1) / dist2
            d2fdsig2dsig3 = (dfdsig3dist3(2) - dfdsig2) / dist3
            d2fdsig3dsig1 = (dfdsig1dist1(3) - dfdsig3) / dist1

            xin(1) = sig_eig_cur(1)
            xin(2) = sig_eig_cur(2)
            xin(3) = sig_eig_cur(3)
            xin(4) = xlamda_cur

            call NN_output(xin, Nlayer_f, NNode_f, layers_f,
     $      ActFun_f, iscalr_f, oscalr_f, WArr_f, BArr_f, fout)

            res(1) = xx(1) - epse_eig_tr(1) + xx(4)*dfdsig1
            res(2) = xx(2) - epse_eig_tr(2) + xx(4)*dfdsig2
            res(3) = xx(3) - epse_eig_tr(3) + xx(4)*dfdsig3
            res(4) = fout(1)
      
            xjac(1,1) = 1.d0 + xx(4)*(d2fdsig1dsig1*dsig1depse1 + d2fdsig1dsig2*dsig2depse1
     $        + d2fdsig3dsig1*dsig3depse1)
            xjac(1,2) =        xx(4)*(d2fdsig1dsig1*dsig1depse2 + d2fdsig1dsig2*dsig2depse2
     $        + d2fdsig3dsig1*dsig3depse2)
            xjac(1,3) =        xx(4)*(d2fdsig1dsig1*dsig1depse3 + d2fdsig1dsig2*dsig2depse3 
     $        + d2fdsig3dsig1*dsig3depse3)
            xjac(1,4) = dfdsig1
      
            xjac(2,1) =        xx(4)*(d2fdsig1dsig2*dsig1depse1 + d2fdsig2dsig2*dsig2depse1
     $        + d2fdsig2dsig3*dsig3depse1)
            xjac(2,2) = 1.d0 + xx(4)*(d2fdsig1dsig2*dsig1depse2 + d2fdsig2dsig2*dsig2depse2
     $        + d2fdsig2dsig3*dsig3depse2)
            xjac(2,3) =        xx(4)*(d2fdsig1dsig2*dsig1depse3 + d2fdsig2dsig2*dsig2depse3 
     $        + d2fdsig2dsig3*dsig3depse3)
            xjac(2,4) = dfdsig2
      
            xjac(3,1) =        xx(4)*(d2fdsig3dsig1*dsig1depse1 + d2fdsig2dsig3*dsig2depse1
     $        + d2fdsig3dsig3*dsig3depse1)
            xjac(3,2) =        xx(4)*(d2fdsig3dsig1*dsig1depse2 + d2fdsig2dsig3*dsig2depse2
     $        + d2fdsig3dsig3*dsig3depse2)
            xjac(3,3) = 1.d0 + xx(4)*(d2fdsig3dsig1*dsig1depse3 + d2fdsig2dsig3*dsig2depse3 
     $        + d2fdsig3dsig3*dsig3depse3)
            xjac(3,4) = dfdsig3
      
            xjac(4,1) = dfdsig1*dsig1depse1 + dfdsig2*dsig2depse1 + dfdsig3*dsig3depse1
            xjac(4,2) = dfdsig1*dsig1depse2 + dfdsig2*dsig2depse2 + dfdsig3*dsig3depse2 
            xjac(4,3) = dfdsig1*dsig1depse3 + dfdsig2*dsig2depse3 + dfdsig3*dsig3depse3
            xjac(4,4) = 0.d0

            call gauss_2(xjac, res, dxx, 4)

            ! Update vector xx
            xx = xx - dxx
            
            ! Compute error
            err = sqrt(dxx(1)**2 + dxx(2)**2 + dxx(3)**2 + dxx(4)**2)
          end do 

          if (kewton .eq. newton+1) then
            write(7,*)'WARNING: plasticity loop failed' 
            print *,'WARNING: plasticity loop failed'
            print *,"error: ", sqrt(dxx(1)**2 + dxx(2)**2 + dxx(3)**2 + dxx(4)**2)
            print *,toler
          end if

          ! Update stress and strain
          eps = eps + deps 
          epse = xx(1)*xm1 + xx(2)*xm2 + xx(3)*xm3 
          epsp = eps - epse 
          xlamda = xlamda + xx(4)
                  
          ! Update stress
          sig = eLam * (epse(1,1) + epse(2,2) + epse(3,3)) * EYE2
     $      + 2.d0*eG*epse   
          
          sig_eig_cur(1) = aa*xx(1) + bb*xx(2) + bb*xx(3)
          sig_eig_cur(2) = bb*xx(1) + aa*xx(2) + bb*xx(3)
          sig_eig_cur(3) = bb*xx(1) + bb*xx(2) + aa*xx(3)

          xin(1) = sig_eig_cur(1)
          xin(2) = sig_eig_cur(2)
          xin(3) = sig_eig_cur(3)
          xin(4) = xlamda_cur

          ! Compute algorithmic tangent
          call NN_output(xin, Nlayer_df, NNode_df, Layers_df, 
     $      ActFun_df, iscalr_df, oscalr_df, WArr_df , BArr_df, df)

          dfdsig1 = df(1)
          dfdsig2 = df(2)
          dfdsig3 = df(3)

          dist1 = 1d-7 * (iscalr_df(2) - iscalr_df(1))
          dist2 = 1d-7 * (iscalr_df(4) - iscalr_df(3))
          dist3 = 1d-7 * (iscalr_df(6) - iscalr_df(5))
          dist4 = 1d-7 * (iscalr_df(8) - iscalr_df(7))

          xindist = xin

          xindist(1) = xindist(1) + dist1
          call NN_output(xindist, Nlayer_df, NNode_df, Layers_df, 
     $      ActFun_df, iscalr_df, oscalr_df, WArr_df, BArr_df, dfdsig1dist1)

          xindist = xin

          xindist(2) = xindist(2) + dist2
          call NN_output(xindist, Nlayer_df, NNode_df, Layers_df, 
     $      ActFun_df, iscalr_df, oscalr_df, WArr_df, BArr_df, dfdsig2dist2)

          xindist = xin

          xindist(3) = xindist(3) + dist3
          call NN_output(xindist, Nlayer_df, NNode_df, Layers_df, 
     $      ActFun_df, iscalr_df, oscalr_df, WArr_df, BArr_df, dfdsig3dist3)

          d2fdsig1dsig1 = (dfdsig1dist1(1) - dfdsig1) / dist1
          d2fdsig2dsig2 = (dfdsig2dist2(2) - dfdsig2) / dist2
          d2fdsig3dsig3 = (dfdsig3dist3(3) - dfdsig3) / dist3

          d2fdsig1dsig2 = (dfdsig2dist2(1) - dfdsig1) / dist2
          d2fdsig2dsig3 = (dfdsig3dist3(2) - dfdsig2) / dist3
          d2fdsig3dsig1 = (dfdsig1dist1(3) - dfdsig3) / dist1

          xjac(1,1) = 1.d0 + xx(4)*(d2fdsig1dsig1*dsig1depse1 + d2fdsig1dsig2*dsig2depse1
     $        + d2fdsig3dsig1*dsig3depse1)
          xjac(1,2) =        xx(4)*(d2fdsig1dsig1*dsig1depse2 + d2fdsig1dsig2*dsig2depse2
     $        + d2fdsig3dsig1*dsig3depse2)
          xjac(1,3) =        xx(4)*(d2fdsig1dsig1*dsig1depse3 + d2fdsig1dsig2*dsig2depse3 
     $        + d2fdsig3dsig1*dsig3depse3)
          xjac(1,4) = dfdsig1
    
          xjac(2,1) =        xx(4)*(d2fdsig1dsig2*dsig1depse1 + d2fdsig2dsig2*dsig2depse1
     $        + d2fdsig2dsig3*dsig3depse1)
          xjac(2,2) = 1.d0 + xx(4)*(d2fdsig1dsig2*dsig1depse2 + d2fdsig2dsig2*dsig2depse2
     $        + d2fdsig2dsig3*dsig3depse2)
          xjac(2,3) =        xx(4)*(d2fdsig1dsig2*dsig1depse3 + d2fdsig2dsig2*dsig2depse3 
     $        + d2fdsig2dsig3*dsig3depse3)
          xjac(2,4) = dfdsig2
    
          xjac(3,1) =        xx(4)*(d2fdsig3dsig1*dsig1depse1 + d2fdsig2dsig3*dsig2depse1
     $        + d2fdsig3dsig3*dsig3depse1)
          xjac(3,2) =        xx(4)*(d2fdsig3dsig1*dsig1depse2 + d2fdsig2dsig3*dsig2depse2
     $        + d2fdsig3dsig3*dsig3depse2)
          xjac(3,3) = 1.d0 + xx(4)*(d2fdsig3dsig1*dsig1depse3 + d2fdsig2dsig3*dsig2depse3 
     $        + d2fdsig3dsig3*dsig3depse3)
          xjac(3,4) = dfdsig3
    
          xjac(4,1) = dfdsig1*dsig1depse1 + dfdsig2*dsig2depse1 + dfdsig3*dsig3depse1
          xjac(4,2) = dfdsig1*dsig1depse2 + dfdsig2*dsig2depse2 + dfdsig3*dsig3depse2 
          xjac(4,3) = dfdsig1*dsig1depse3 + dfdsig2*dsig2depse3 + dfdsig3*dsig3depse3
          xjac(4,4) = 0.d0

          do i = 1,3
            do j = 1,3
              xjac_tmp(i,j) = xjac(i,j)
            end do
          end do
          
          call matrixinv(xjac_tmp, xjac_tmpinv, 3)
          aAB(1,1) = dsig1depse1 * xjac_tmpinv(1,1) + dsig1depse2 * xjac_tmpinv(2,1) + dsig1depse3 * xjac_tmpinv(3,1) 
          aAB(1,2) = dsig1depse1 * xjac_tmpinv(1,2) + dsig1depse2 * xjac_tmpinv(2,2) + dsig1depse3 * xjac_tmpinv(3,2) 
          aAB(1,3) = dsig1depse1 * xjac_tmpinv(1,3) + dsig1depse2 * xjac_tmpinv(2,3) + dsig1depse3 * xjac_tmpinv(3,3) 
          
          aAB(2,1) = dsig2depse1 * xjac_tmpinv(1,1) + dsig2depse2 * xjac_tmpinv(2,1) + dsig2depse3 * xjac_tmpinv(3,1) 
          aAB(2,2) = dsig2depse1 * xjac_tmpinv(1,2) + dsig2depse2 * xjac_tmpinv(2,2) + dsig2depse3 * xjac_tmpinv(3,2) 
          aAB(2,3) = dsig2depse1 * xjac_tmpinv(1,3) + dsig2depse2 * xjac_tmpinv(2,3) + dsig2depse3 * xjac_tmpinv(3,3) 
          
          aAB(3,1) = dsig3depse1 * xjac_tmpinv(1,1) + dsig3depse2 * xjac_tmpinv(2,1) + dsig3depse3 * xjac_tmpinv(3,1) 
          aAB(3,2) = dsig3depse1 * xjac_tmpinv(1,2) + dsig3depse2 * xjac_tmpinv(2,2) + dsig3depse3 * xjac_tmpinv(3,2) 
          aAB(3,3) = dsig3depse1 * xjac_tmpinv(1,3) + dsig3depse2 * xjac_tmpinv(2,3) + dsig3depse3 * xjac_tmpinv(3,3) 

          do i = 1,3 
            do j = 1,3 
              xm11(i,j) = eigvctmp(i,1) * eigvctmp(j,1)
              xm22(i,j) = eigvctmp(i,2) * eigvctmp(j,2)
              xm33(i,j) = eigvctmp(i,3) * eigvctmp(j,3)
              
              xm12(i,j) = eigvctmp(i,1) * eigvctmp(j,2)
              xm21(i,j) = eigvctmp(i,2) * eigvctmp(j,1)

              xm23(i,j) = eigvctmp(i,2) * eigvctmp(j,3)              
              xm32(i,j) = eigvctmp(i,3) * eigvctmp(j,2)

              xm13(i,j) = eigvctmp(i,1) * eigvctmp(j,3)
              xm31(i,j) = eigvctmp(i,3) * eigvctmp(j,1)
            end do
          end do

          if ( ( epse_eig_tr(2) - epse_eig_tr(1) ) .le. toler ) then
            term12 = 0.d0
          else
            term12 = ( sig_eig_cur(2) - sig_eig_cur(1) )/( epse_eig_tr(2) - epse_eig_tr(1) )
          endif
          if ( ( epse_eig_tr(3) - epse_eig_tr(1) ) .le. toler ) then
            term13 = 0.d0
          else
            term13 = ( sig_eig_cur(3) - sig_eig_cur(1) )/( epse_eig_tr(3) - epse_eig_tr(1) )
          endif
          
          if ( ( epse_eig_tr(1) - epse_eig_tr(2) ) .le. toler ) then
            term21 = 0.d0
          else
            term21 = ( sig_eig_cur(1) - sig_eig_cur(2) )/( epse_eig_tr(1) - epse_eig_tr(2) )
          endif
          if ( ( epse_eig_tr(3) - epse_eig_tr(2) ) .le. toler ) then
            term23 = 0.d0
          else
            term23 = ( sig_eig_cur(3) - sig_eig_cur(2) )/( epse_eig_tr(3) - epse_eig_tr(2) )
          endif
          
          if ( ( epse_eig_tr(1) - epse_eig_tr(3) ) .le. toler ) then
            term31 = 0.d0
          else
            term31 = ( sig_eig_cur(1) - sig_eig_cur(3) )/( epse_eig_tr(1) - epse_eig_tr(3) )
          endif
          if ( ( epse_eig_tr(2) - epse_eig_tr(3) ) .le. toler ) then
            term32 = 0.d0
          else
            term32 = ( sig_eig_cur(2) - sig_eig_cur(3) )/( epse_eig_tr(2) - epse_eig_tr(3) )
          endif
          
          C_algo = 0.d0
          do i = 1,3 
            do j = 1,3
              do k = 1,3
                do l = 1,3
         C_algo(i,j,k,l) = aAB(1,1) * xm11(i,j)*xm11(k,l) + aAB(1,2) * xm11(i,j)*xm22(k,l) +  aAB(1,3) * xm11(i,j)*xm33(k,l)
     $                   + aAB(2,1) * xm22(i,j)*xm11(k,l) + aAB(2,2) * xm22(i,j)*xm22(k,l) +  aAB(2,3) * xm22(i,j)*xm33(k,l)
     $                   + aAB(3,1) * xm33(i,j)*xm11(k,l) + aAB(3,2) * xm33(i,j)*xm22(k,l) +  aAB(3,3) * xm33(i,j)*xm33(k,l) 
     $             + 0.5d0*term12*(xm12(i,j)*xm12(k,l) + xm12(i,j)*xm21(k,l))
     $             + 0.5d0*term13*(xm13(i,j)*xm13(k,l) + xm13(i,j)*xm31(k,l))
     $             + 0.5d0*term21*(xm21(i,j)*xm21(k,l) + xm21(i,j)*xm12(k,l))
     $             + 0.5d0*term23*(xm23(i,j)*xm23(k,l) + xm23(i,j)*xm32(k,l))
     $             + 0.5d0*term31*(xm31(i,j)*xm31(k,l) + xm31(i,j)*xm13(k,l))
     $             + 0.5d0*term32*(xm32(i,j)*xm32(k,l) + xm32(i,j)*xm23(k,l))
                end do
              end do
            end do
          end do

          call TensorTovoigt(C_algo, ddsdde)
        end if
        ! ==================== STRESS INTEGRATION ENDS ==================== !
        
        ! Update stress and statev
        stress(1) = sig(1,1)
        stress(2) = sig(2,2)
        stress(3) = sig(3,3)
        stress(4) = sig(1,2)
        stress(5) = sig(1,3)
        stress(6) = sig(2,3)

        statev(1) = epse(1,1)
        statev(2) = epse(2,2)
        statev(3) = epse(3,3)
        statev(4) = epse(1,2) * 2.d0
        statev(5) = epse(1,3) * 2.d0
        statev(6) = epse(2,3) * 2.d0
        
        statev(7)  = epsp(1,1)
        statev(8)  = epsp(2,2)
        statev(9)  = epsp(3,3)
        statev(10) = epsp(1,2) * 2.d0
        statev(11) = epsp(1,3) * 2.d0
        statev(12) = epsp(2,3) * 2.d0
        
        statev(13) = xlamda

      end subroutine umat
C =================================================================== C