# My first hosted observation stops during setup

Run 46051728142/job140684444894, observer 31e4b6a275, fails before loading my
observer or running calibration or Stage2. My sparse checkout omits the retained
capture-profile manifest. Two of the eight existing capture setup controls
raise FileNotFoundError. The later observation-artifact upload also finds no
files because setup stopped before my output directory existed. I preserve the
complete job API/log and the two-member capture-control artifact.

I restore exactly the canonical manifest and sixteen profile inputs from the
pinned d18 Git tree. I verify compressed and decoded hashes and lengths before
writing each input, including objects referenced in sibling evidence archives.
I preserve the original Python prerequisite and create checkout evidence before
provisioning so another setup refusal is retained.

I execute the exact workflow materializer on a separate d18 source checkout,
then run the original eight setup controls: all pass. I read the retained x86
ELF as data on Sparky; I do not execute it as a native assembler here. I do not
change the original capture tests or any product assertion or limit. Corrected
hosted admission and actual Stage2 observation remain pending.
