# Makefile for gres/opencl plugin

AUTOMAKE_OPTIONS = foreign

PLUGIN_FLAGS = -module -avoid-version --export-dynamic

AM_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src/common

pkglib_LTLIBRARIES = gres_opencl.la

# Gres GPU plugin.
gres_opencl_la_SOURCES = gres_opencl.c
gres_opencl_la_LDFLAGS = $(PLUGIN_FLAGS)
gres_opencl_la_LIBADD = ../common/libgres_common.la

force:
$(gres_opencl_la_LIBADD) : force
	@cd `dirname $@` && $(MAKE) `basename $@`
