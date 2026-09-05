#  Preliminary Test Program #

##  Rationale ##

The Hayes tester (`tester.fr`) for ANS Forth has the drawback that it needs a lot  of a Forth system to be working before the test programs can be run. This detracts from the progressive nature of the Hayes core tests where words are tested before being used. The Hayes tester requires the following to be fully working:

- 32 words from the Core word set
- 4 words from the Core Extension word set
- some rather complex colon definitions
- the Forth word `INCLUDED` or equivalent (not essential but desirable)

This is perfectly acceptable for a working system being re-tested but not for a new system under development where it is likely that the tester itself has to be debugged before testing can proceed. What is needed for such a system is a pre-Hayes test bootstrap program that assumes as little as possible of the system is proven and which tests, in a rudimentary way if not exhaustively, all the words used in the tester. The test file `prelimtest.fth` provides such a test program.

##  Minimal system requirements for automated testing ##

At the very least for manual or automated testing, the Forth outer text interpreter has to be working which implies that the system can:

-    accept a line of input from the keyboard/stdin
-    display some text on the system monitor
-    parse a line of text into space delimited words, this implies that Forth words `>IN` and `WORD` work and that `SOURCE` is available
-    search the Forth dictionary for a word, hence `FIND` or a factor of `FIND` works
-    if the word is found execute it
-    otherwise recognise it as a valid integer and convert it using `BASE`
-    otherwise report an 'undefined word' error or fail in some other way
-    have working stacks (both data and return)
 
To avoid tests being typed in it helps if `INCLUDED` works so that a file can be interpreted. That is not essential as input can be redirected from a file or pasted in to the command window but in the latter cases it can be difficult to inspect output from the tests due to the interpreted text being displayed between test results. A working `INCLUDED` is recommended. This file is the documentation for the test program.

##  Assumptions for the pre-Hayes tester ##

1. The outer text interpreter must be functional using at least input from a keyboard/stdin or, preferably a file using INCLUDED or similar. In particular the words `SOURCE` `TYPE` and `CR` need to work for the initial tests. 
2. All core words used in the Hayes tester are available for testing. 
3. In the early tests success will be verified by the user inspecting a 'Pass' message on the  system display until such time as sufficient words have been defined and tested to enable  automated testing. 
4. If the early tests fail the Forth system reaction is unpredictable and will probably do at least one of: 
	- 	display an error message e.g. 'undefined word'
	- 	crash in some way e.g. enter an infinite loop, access an illegal address, stop running
	- 	ignore the failure and carry on so that the user can see that the 'Pass' message is missing.
1. The system will only display the 'Pass' message if the test has actually passed.
2. Tests on standard words will be rudimentary rather than exhaustive, just testing the basic operation.
2. Two's complement arithmetic is assumed.

## Restrictions on the tester ##

1. Only words from the Core word set and will be used in the tester, any others will have to be defined unless already defined - this particularly applies to the four Core Extension words used in the Hayes tester. This restriction unfortunately rules out the use of `\` `.(` `PARSE` and a few other useful words.
2. Only absolutely necessary words will be defined.
3. Check that true and false flags are all 1's or 0's respectively.
4. Make the tests progressive i.e. test before use.


## The test sequence ##

1. The first tests simply reproduce the line of Forth code, if they display correctly then the test has passed, there is no 'Pass' message.
2. The next few tests are checked by visual inspection of the display which, if passed, will display a line beginning 'Pass #nn: testing xyz' where nn is a sequential number used to identify failures. The first few tests have such a message in parentheses.
3. As soon as possible a word is defined that checks the test result and reports errors rather than relying on visual inspection. To avoid having to use `IF` the check manipulates `>IN` to implement an interpretive if.
4. The error reporting from point 3 is used to test the rest of the words used in the Hayes tester.
5. If the four Core Extension words are not available they are defined.
6. An error report is provided at the end.

## Test Output ##

This is given in file `doc/prelimtestoutput.txt`. Other system generated messages may be displayed e.g. redefinition messages. 

## Notes ##

If the system under test cannot include files then some way will have to be found to input the test programs into the system. This is system dependent and hence not considered further here.

Initial testing of a result is done by the trick of incrementing `>IN`, a typical example is:
 
     <some Forth code returning 0 or 1> >IN +! xSOURCE TYPE
 
where 0 indicates failure and 1 success. If 0 then `>IN` is not incremented and the system tries to find `xSOURCE` and (hopefully) fails with an 'undefined word' message. If 1 then >IN is incremented and the word `SOURCE` is found and executed and the line displayed. Several variations of this theme are used in the test program. If the Forth code returns some other number then `>IN` may index outside the boundaries of the input buffer and the resulting behaviour is unpredictable. In the early tests it is likely that the pass message for the test will not be displayed and the error detected by the user. After = is proved to return a well-formed flag and the error reporting word defined and used, this problem no longer exists.
 
No assumptions have been made about the value held in BASE and integers bigger than 1 can't initially be used without possible error. Therefore BASE is set to 2 using the tested 1+ and then decimal 10 by using its binary value. The system is then in decimal mode and digits > 1 can be safely used.

An error reporting word is needed, `.(` and `PARSE` would be useful except that they are in the Core Extension word set, only `WORD` is available to parse a message and must be used in a colon definition. Therefore simple colon definitions are tested by simply defining `.SRC` to display the line of source code.

A word called `MSG` is defined to test `WORD` and `COUNT`, note that decimal 41 is the ASCII code for character ')'. When tested the address returned by `WORD` is system dependent and cannot be tested, is dropped.

A word `.MSG(` is defined to display parsed text thus removing the need to display the whole line of source code, reducing display noise. This is used until the error reporting word has been defined.

Other notes are included in the test program itself.

The final error report uses some low-level tricks to avoid having to use additional Forth words, this avoids the need to test such words before use.

Once a system passes this initial test program and runs `tester.fr` it can be removed from the script running the test programs. 

