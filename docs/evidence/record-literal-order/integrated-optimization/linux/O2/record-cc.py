import json, os, sys
with open(os.environ["NANO_LITERAL_CC_LOG"], "a") as out:
    out.write(json.dumps(sys.argv[1:]) + "\n")
command = ['cc'] + sys.argv[1:]
os.execvp(command[0], command)
