SUFFIXES+=.mk
LINK = -lboost_filesystem -lboost_system

ifeq ($(OS),Windows_NT)
	GPP:=g++
else
	GPP:=g++-5
endif


ifeq ($(RELEASE),YES)
	OPTIMIZATION:=-O2
else
	OPTIMIZATION:=-O0 -g -fno-inline -fno-omit-frame-pointer -Wall -Wpedantic -Wextra -Werror -Wno-deprecated-declarations
	ifneq ($(OS),Windows_NT)
		OPTIMIZATION:=$(OPTIMIZATION) -fsanitize=address
	endif
endif


OPTS_SHARED:=-I /usr/include/eigen3 -I ./include -std=c++1z


ifeq ($(OS),Windows_NT)
	MAKE_PARENT=mkdir.bat $(subst /,\,$(dir $(1)))
	RMDIR=rmdir.bat $(subst /,\,$(1))
	MKDIR=mkdir.bat $(subst /,\,$(1))
	MODULE_EXT:=.dll
	EXECUTABLE_EXT:=.exe
else
	MAKE_PARENT=mkdir -p $(dir $(1))
	RMDIR=rm -r -f $(1)
	MKDIR=mkdir $(1)
	OPTS_SHARED:=$(OPTS_SHARED) -fPIC
	MODULE_EXT:=.so
	EXECUTABLE_EXT:=
	LINK += `pkg-config --libs opencv`
endif


GPP:=$(GPP) $(OPTS_SHARED) $(OPTIMIZATION)


MODULE_NAME:=seng499


.PHONY: all
all:


.PHONY: clean
clean:
	-@$(call RMDIR,obj)
	-@$(call RMDIR,bin)
	-@$(call RMDIR,makefiles)


bin:
	-@$(call MKDIR,bin)

bin/modules: | bin
	-@$(call MKDIR,bin/modules)


NODEPS:=clean cleanall cleandeps


ifeq (0,$(words $(findstring $(MAKECMDGOALS),$(NODEPS))))

	ifeq ($(OS),Windows_NT)
		-include $(subst .cpp,.mk,$(subst src/,makefiles/,$(subst \,/,$(subst $(shell echo %CD%)\,,$(shell dir /b /s src\*.cpp)))))
	else
		-include $(patsubst src/%.cpp,makefiles/%.mk,$(shell find src/ -name "*.cpp"))
	endif

endif
	
	
obj/%.o:
	-@$(call MAKE_PARENT,$(patsubst obj/%.o,makefiles/%.mk,$@))
	$(GPP) -MM -MT "$@" $(patsubst obj/%.o,src/%.cpp,$@) -MF $(patsubst obj/%.o,makefiles/%.mk,$@)
	-@$(call MAKE_PARENT,$@)
	$(GPP) -c -o $@ $(patsubst obj/%.o,src/%.cpp,$@)

	
OBJS:=\
obj/depth_device.o \
obj/file_system_depth_device.o \
obj/fps_depth_device.o \
obj/measurement_pipeline_block.o \
obj/mock_depth_device.o \
obj/msrc_file_system_depth_device.o \
obj/kinect_fusion.o \
obj/pose_estimation_pipeline_block.o \
obj/surface_prediction_pipeline_block.o \
obj/update_reconstruction_pipeline_block.o


TEST_OBJS:=\
obj/test/file_system_depth_device.o \
obj/test/msrc_file_system_depth_device.o \
obj/test/fps_depth_device.o


all: bin/$(MODULE_NAME)$(MODULE_EXT)
bin/$(MODULE_NAME)$(MODULE_EXT): \
$(OBJS) | \
bin
	$(GPP) -shared -o $@ $^ $(LINK)


all: bin/tests$(EXECUTABLE_EXT)
bin/tests$(EXECUTABLE_EXT): \
$(TEST_OBJS) \
obj/test/main.o | \
bin \
bin/$(MODULE_NAME)$(MODULE_EXT)
	$(GPP) -o $@ $^ $(LINK) bin/$(MODULE_NAME)$(MODULE_EXT)
	bin/tests$(EXECUTABLE_EXT)
