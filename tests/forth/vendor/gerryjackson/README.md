# forth2012-test-suite
Test programs for Forth 2012 and ANS Forth

Checking that a Forth 2012/ANS Forth system is compliant with the Forth 2012
and ANS Forth standards requires a set of test programs. This repository
provides some test programs that go some way towards meeting this need. The
tests are based on the widely known ANS Forth tester developed by John Hayes
and include his original core tests. The tests cover most of the word sets of
the original ANS Forth (Forth 94) standard and the Forth 2012 standard that
extends and supersedes ANS Forth.

The word sets tested are:
- Core words (by John Hayes) plus some additional tests
- Core extension words
- The Block word set and extensions (see the warning below)
- The Double Number word set and extensions
- The Exception word set and extensions
- The Facility word set extensions (structures only)
- The File-Access word set and extensions
- The Locals word set
- The Memory-Allocation word set
- The Programming-Tools word set and extensions (incomplete - a
  few words in each)
- The Search-Order word set and extensions
- The String word set

**Block test warning - in the file blocktest.fth the block numbers tested are in the range 20 to 29 and will be overwritten when the test is run. These block numbers can be changed at the start of the file.**

The master branch has minor updates to release 0.13 (27 Nov 2015) of the test suite.

The tests are not comprehensive, no claim is made about their being correct and no warranty is provided. However previous versions have been run successfully with such widely used systems as GForth, VFX Forth, SwiftForth and Win32 Forth, any failing tests being due to the Forth system rather than the tests themselves. Any bugs in the tests should be reported by raising a new issue.

**Running the tests**
 1. Unzip the files into a suitable directory
 2. Start the Forth system and set the working directory/file path to that directory
 3. If necessary edit the file paths in the file runtests.fth
 4. Select either tester.fr or ttester.fs by commenting out the unwanted file. ( Use  tester.fr for system development).
 5. Include runtests.fth
 6. Examine screen output.

(Note that the above assumes the system can include source files via the word INCLUDED. If that is not the case then another way of streaming input from the file will have to be found.)

There is a flag, called VERBOSE, in file tester.fr that can be set to obtain more output from the test programs.

When a test fails an error message is displayed and the tests carry on. To stop on the first failure uncomment the line marked *** in tester.fr 

Typical output from the tests is given in the file doc/testoutput.txt. Some tests, particularly those displaying text cannot be automatically checked and require visual inspection.

**Error Messages**
 There are two possible error messages when a test fails:
  1. "WRONG NUMBER OF RESULTS:" when the stack depth of the test result is wrong
  2. "INCORRECT RESULT:" when the stack contents are incorrect

These are followed by the offending line of source code. 

Other failures such as a standard word not being recognised will likely result in the test run terminating. This is the case if you are using
GForth 0.7.3 to run the tests when a message such as:
```
coreexttest.fth:400: Undefined word
T{ 8 >>>BUFFER:<<< BUF:TEST -> }T
etc
```
and the tests will terminate.

This is because BUFFER: is a Forth 2012 addition to ANS Forth (1994) and is not included in GForth 0.7.3 which
was released in 2008. The same is true for all additional words in Forth 2012.

To overcome this please use a recent snapshot executable or compile and build Gforth yourself from:
[GForth 0.7.9](https://www.complang.tuwien.ac.at/forth/gforth/Snapshots/)

**Floating Point Tests**
Some floating point test programs from various sources have been collected together and made available in this download. For more details see the file fp/readme-fp.txt in the download. These tests are unproven and provided in the hope that implementers of ANS Forth/Forth 200X systems will try them and report back on any perceived deficiencies which can then be resolved. When they are generally agreed to be correct they can be incorporated into the above set of test programs. (This would seem to be extremely optimistic as only one communication has been received in nearly four years!)
