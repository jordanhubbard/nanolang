Floating Point Test Programs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following floating point test programs are included in this
download:

1. ak-fp-test.fth by A.K. (full name unknown) with some system
   dependent tests removed.
2. fatan2-test.fs by David N Williams
3. ieee-arith-test.fs by David N Williams
4. ieee-fprox-test.fs by David N Williams
5. fpzero-test.4th by Krishna Myneni
6. fpio-test.4th by Krishna Myneni
7. to-float-test.4th by Krishna Myneni
8. paranoia.4th converted by Krishna Myneni (see the file for
   the original source)

In addition:

9. ttester.fs an extended version of the Hayes tester written
   (see the file for authors)
10. runfptests.fth to load and run all the above tests

Running the tests
~~~~~~~~~~~~~~~~~

Unzip into a convenient directory, make that the working
directory and type
      s" runfptests.fth" included
or similar.

Notes
~~~~~

1. These tests are unproven on various Forth systems. They run
without errors on 32 bit GForth 0.7.0 but with one flaw
reported by paranoia.4th.

2. The only way in which the tests will be improved is for
Forth system implementers to use the test programs and feedback
perceived deficiencies or errors. These may then be resolved by
discussion and the test programs amended.

3. Please feedback comments/complaints etc to:
        gerry@jackson9000.fsnet.co.uk
   or post them on the Forth 200X discussion group:
        http://tech.groups.yahoo.com/group/forth200x/
