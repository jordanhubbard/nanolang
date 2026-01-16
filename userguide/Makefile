.PHONY: html check serve clean

ROOT := ..
OUT_DIR := $(ROOT)/build/userguide/html
SERVER_BIN := $(ROOT)/build/userguide/serve_userguide

html:
	python3 $(ROOT)/scripts/userguide_build_html.py

check:
	python3 $(ROOT)/scripts/userguide_snippets_check.py

serve: html
	$(ROOT)/bin/nanoc serve.nano -o $(SERVER_BIN)
	$(SERVER_BIN)

clean:
	rm -rf $(ROOT)/build/userguide
