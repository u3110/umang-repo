!     -*- f90 -*-
!     This file is autogenerated with f2py (version:2)
!     It contains Fortran 90 wrappers to fortran functions.

      
      subroutine f2pyinithalftone(f2pysetupfunc)
      use halftone, only : shiaufan_iterator
      use halftone, only : atkinson_iterator
      use halftone, only : sierra24a_iterator
      use halftone, only : sierra2_iterator
      use halftone, only : sierra3_iterator
      use halftone, only : burkes_iterator
      use halftone, only : stucki_iterator
      use halftone, only : floyd_steinberg_iterator
      use halftone, only : jarvis_iterator
      use halftone, only : ordered_dithering3x3
      use halftone, only : ordered_comb4_iterator
      use halftone, only : ordered_comb2_iterator
      use halftone, only : ordered_comb3_iterator
      external f2pysetupfunc
      call f2pysetupfunc(shiaufan_iterator,atkinson_iterator,sierra24a_i&
     &terator,sierra2_iterator,sierra3_iterator,burkes_iterator,stucki_i&
     &terator,floyd_steinberg_iterator,jarvis_iterator,ordered_dithering&
     &3x3,ordered_comb4_iterator,ordered_comb2_iterator,ordered_comb3_it&
     &erator)
      end subroutine f2pyinithalftone

      
      subroutine f2pyinitinverse_halftone(f2pysetupfunc)
      use inverse_halftone, only : inverse_ordered_comb2_iterator
      use inverse_halftone, only : inverse_ordered_comb3_iterator
      use inverse_halftone, only : inverse_ordered_dithering_iterator
      external f2pysetupfunc
      call f2pysetupfunc(inverse_ordered_comb2_iterator,inverse_ordered_&
     &comb3_iterator,inverse_ordered_dithering_iterator)
      end subroutine f2pyinitinverse_halftone

      subroutine f2pywrap_fbih_getpixel (getpixelf2pywrap, pos, blocksiz&
     &e, blk)
      use fbih, only : getpixel
      integer pos
      integer blocksize
      real blk(blocksize)
      real getpixelf2pywrap
      getpixelf2pywrap = getpixel(pos, blocksize, blk)
      end subroutine f2pywrap_fbih_getpixel
      subroutine f2pywrap_fbih_reflect (reflectf2pywrap, x, y)
      use fbih, only : reflect
      integer x
      integer y
      integer reflectf2pywrap
      reflectf2pywrap = reflect(x, y)
      end subroutine f2pywrap_fbih_reflect
      subroutine f2pywrap_fbih_bitblockcounter (bitblockcounterf2pywrap,&
     & rows, cols, row, col, img)
      use fbih, only : bitblockcounter
      integer rows
      integer cols
      integer row
      integer col
      real img(rows,cols)
      integer bitblockcounterf2pywrap
      bitblockcounterf2pywrap = bitblockcounter(rows, cols, row, col, im&
     &g)
      end subroutine f2pywrap_fbih_bitblockcounter
      
      subroutine f2pyinitfbih(f2pysetupfunc)
      use fbih, only : inverse_halftone
      use fbih, only : medianfilter3by3
      use fbih, only : swap
      use fbih, only : gaussianfilter1
      use fbih, only : separable9x9firbinaryimage
      use fbih, only : rowconvolutions9by9
      use fbih, only : columnconvolutions9by9
      use fbih, only : gaussianfilter2
      use fbih, only : gaussianfilter3
      use fbih, only : separable7x7firgreyimage
      use fbih, only : columnconvolutions7by7
      use fbih, only : thresholddiff
      use fbih, only : binarymedialfilter5by5
      use fbih, only : normalise
      interface 
      subroutine f2pywrap_fbih_getpixel (getpixelf2pywrap, getpixel, pos&
     &, blocksize, blk)
      real getpixel
      integer pos
      integer blocksize
      real blk(blocksize)
      real getpixelf2pywrap
      end subroutine f2pywrap_fbih_getpixel 
      subroutine f2pywrap_fbih_reflect (reflectf2pywrap, reflect, x, y)
      integer reflect
      integer x
      integer y
      integer reflectf2pywrap
      end subroutine f2pywrap_fbih_reflect 
      subroutine f2pywrap_fbih_bitblockcounter (bitblockcounterf2pywrap,&
     & bitblockcounter, rows, cols, row, col, img)
      integer bitblockcounter
      integer rows
      integer cols
      integer row
      integer col
      real img(rows,cols)
      integer bitblockcounterf2pywrap
      end subroutine f2pywrap_fbih_bitblockcounter
      end interface
      external f2pysetupfunc
      call f2pysetupfunc(inverse_halftone,medianfilter3by3,swap,f2pywrap&
     &_fbih_getpixel,gaussianfilter1,separable9x9firbinaryimage,rowconvo&
     &lutions9by9,columnconvolutions9by9,gaussianfilter2,gaussianfilter3&
     &,separable7x7firgreyimage,columnconvolutions7by7,f2pywrap_fbih_ref&
     &lect,thresholddiff,binarymedialfilter5by5,f2pywrap_fbih_bitblockco&
     &unter,normalise)
      end subroutine f2pyinitfbih


