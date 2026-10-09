# My explicit SDL image imports

I retain both strict-example failures from GitHub Actions run 37931895899 at 5dc3a17fb. A temporary diagnostic names SDL_BLENDMODE_NONE as the conflicting public global imported from SDL and SDL_image. I preserve the conflict check and import image functions by namespace, with IMG_INIT_PNG explicitly public and selectively imported.

My integrated launcher compiles with `bin/nanoc_c examples/sdl_example_launcher.nano -o /private/tmp/nanolang-launcher-global-candidate/integrated-launcher`. I did not run the GUI. My two earlier temporary probes fail generated C compilation; the selective probe omitted the original module metadata and is not evidence of an additional product defect. I retain those terminals separately. The temporary diagnostic is not integrated.

Fresh hosted strict-example qualification remains required under #986.
