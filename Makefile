CC = arm-none-eabi-gcc
BUILD_DIR = output
NWLINK = npx --yes -- nwlink@0.0.19

CFLAGS = -std=c99 $(shell $(NWLINK) eadk-cflags-device) -Os -Wall -fdata-sections -ffunction-sections
LDFLAGS = -Wl,--relocatable -nostartfiles --specs=nano.specs -Wl,-e,main -Wl,-u,eadk_app_name -Wl,-u,eadk_app_icon -Wl,-u,eadk_api_level -Wl,--gc-sections -lm

all: $(BUILD_DIR)/NumLLM.nwa

$(BUILD_DIR)/NumLLM.nwa: $(BUILD_DIR)/main.o $(BUILD_DIR)/icon.o
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@
	$(NWLINK) nwa-bin $@ $(BUILD_DIR)/NumLLM.bin
	@ls -lh $@

$(BUILD_DIR)/main.o: src/main.c src/model_weights.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c src/main.c -o $@

$(BUILD_DIR)/icon.o: src/icon.png | $(BUILD_DIR)
	$(NWLINK) png-icon-o $< $@

$(BUILD_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)
