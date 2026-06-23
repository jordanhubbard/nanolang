.PHONY: html check serve clean

ROOT := ..
OUT_DIR := $(ROOT)/build/userguide/html
SERVER_BIN := $(ROOT)/build/userguide/serve_userguide

html:
	$(MAKE) -C $(ROOT) userguide-html

check:
	$(MAKE) -C $(ROOT) userguide-check

serve: html
	cd $(ROOT) && ./bin/nanoc userguide/serve.nano -o build/userguide/serve_userguide
	cd $(ROOT) && ./build/userguide/serve_userguide

clean:
	rm -rf $(ROOT)/build/userguide
