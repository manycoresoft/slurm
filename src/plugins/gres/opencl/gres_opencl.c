/*****************************************************************************\
 *  gres_opencl.c - Support OpenCL devices as a generic resources.
 *****************************************************************************
 *  Copyright (C) 2018 ManyCoreSoft Co., Ltd.
 *  Written by Gangwon Jo <gangwon@manycoresoft.co.kr>
 *  Based upon gres_gpu.c with the copyright notice shown below:
 *  Copyright (C) 2010 Lawrence Livermore National Security.
 *  Produced at Lawrence Livermore National Laboratory (cf, DISCLAIMER).
 *  Written by Morris Jette <jette1@llnl.gov>
 *
 *  This file is part of Slurm, a resource management program.
 *  For details, see <https://slurm.schedmd.com/>.
 *  Please also read the included file: DISCLAIMER.
 *
 *  Slurm is free software; you can redistribute it and/or modify it under
 *  the terms of the GNU General Public License as published by the Free
 *  Software Foundation; either version 2 of the License, or (at your option)
 *  any later version.
 *
 *  In addition, as a special exception, the copyright holders give permission
 *  to link the code of portions of this program with the OpenSSL library under
 *  certain conditions as described in each individual source file, and
 *  distribute linked combinations including the two. You must obey the GNU
 *  General Public License in all respects for all of the code used other than
 *  OpenSSL. If you modify file(s) with this exception, you may extend this
 *  exception to your version of the file(s), but you are not obligated to do
 *  so. If you do not wish to do so, delete this exception statement from your
 *  version.  If you delete this exception statement from all source files in
 *  the program, then also delete it here.
 *
 *  Slurm is distributed in the hope that it will be useful, but WITHOUT ANY
 *  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 *  FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 *  details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with Slurm; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA.
\*****************************************************************************/

#define _GNU_SOURCE

#include <ctype.h>
#include <inttypes.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include "slurm/slurm.h"
#include "slurm/slurm_errno.h"

#include "src/common/slurm_xlator.h"
#include "src/common/bitstring.h"
#include "src/common/env.h"
#include "src/common/gres.h"
#include "src/common/list.h"
#include "src/common/xcgroup_read_config.c"
#include "src/common/xstring.h"

#include "../common/gres_common.h"

/*
 * These variables are required by the generic plugin interface.  If they
 * are not found in the plugin, the plugin loader will ignore it.
 *
 * plugin_name - A string giving a human-readable description of the
 * plugin.  There is no maximum length, but the symbol must refer to
 * a valid string.
 *
 * plugin_type - A string suggesting the type of the plugin or its
 * applicability to a particular form of data or method of data handling.
 * If the low-level plugin API is used, the contents of this string are
 * unimportant and may be anything.  Slurm uses the higher-level plugin
 * interface which requires this string to be of the form
 *
 *	<application>/<method>
 *
 * where <application> is a description of the intended application of
 * the plugin (e.g., "auth" for Slurm authentication) and <method> is a
 * description of how this plugin satisfies that application.  Slurm will
 * only load authentication plugins if the plugin_type string has a prefix
 * of "auth/".
 *
 * plugin_version - an unsigned 32-bit integer containing the Slurm version
 * (major.minor.micro combined into a single number).
 */
const char plugin_name[] = "Gres OpenCL plugin";
const char plugin_type[] = "gres/opencl";
const uint32_t plugin_version = SLURM_VERSION_NUMBER;

static char gres_name[] = "opencl";
static List gres_devices = NULL;
static int gres_device_num = 0;

static void _set_env(char ***env_ptr, void *gres_ptr, int node_inx,
		     bitstr_t *usable_gres,
		     bool *already_seen, bool reset, bool is_job)
{
	char *device_list = NULL;
	bitstr_t *bit_alloc = NULL;
	int i, len;
	gres_device_t *gres_device, *first_device = NULL;
	ListIterator itr;

	if (!gres_devices)
		return;
	if (reset && !usable_gres)
		return;

	if (is_job) {
		gres_job_state_t *gres_job_ptr = (gres_job_state_t *) gres_ptr;
		if (gres_job_ptr &&
		    (node_inx >= 0) &&
		    (node_inx < gres_job_ptr->node_cnt) &&
		    gres_job_ptr->gres_bit_alloc &&
		    gres_job_ptr->gres_bit_alloc[node_inx]) {
			bit_alloc = gres_job_ptr->gres_bit_alloc[node_inx];
		}
	} else {
		gres_step_state_t *gres_step_ptr = (gres_step_state_t *) gres_ptr;
		if (gres_step_ptr &&
		    (gres_step_ptr->node_cnt == 1) &&
		    gres_step_ptr->gres_bit_alloc &&
		    gres_step_ptr->gres_bit_alloc[0]) {
			bit_alloc = gres_step_ptr->gres_bit_alloc[0];
		}
	}

	if (!bit_alloc) {
		debug("%s: unable to set env vars, no device files configured",
		      __func__);
		return;
	}

	len = bit_size(bit_alloc);
	if (len != list_count(gres_devices)) {
		error("%s: gres list is not equal to the number of gres_devices.  This should never happen.",
		      __func__);
		return;
	}

	if (*already_seen) {
		device_list = xstrdup(getenvp(*env_ptr,
		                              "OPENCL_VISIBLE_DEVICES"));
	}

	i = -1;
	itr = list_iterator_create(gres_devices);
	while ((gres_device = list_next(itr))) {
		i++;
		if (!bit_test(bit_alloc, i))
			continue;
		if (reset) {
			if (!first_device)
				first_device = gres_device;
			if (!bit_test(usable_gres, i))
				continue;
		}
		if (device_list) {
			xstrcat(device_list, "::");
		}
		xstrfmtcat(device_list, "%s:%s",
		           gres_device->path, gres_device->major);
	}
	list_iterator_destroy(itr);

	if (reset && !device_list && first_device) {
		xstrfmtcat(device_list, "%s:%s",
		           first_device->path, first_device->major);
	}

	if (device_list) {
		env_array_overwrite(env_ptr, "OPENCL_VISIBLE_DEVICES",
		                    device_list);
		xfree(device_list);
		*already_seen = true;
	}
}

extern int init(void)
{
	debug("%s: %s loaded", __func__, plugin_name);
	return SLURM_SUCCESS;
}

extern int fini(void)
{
	debug("%s: unloading %s", __func__, plugin_name);
	FREE_NULL_LIST(gres_devices);
	return SLURM_SUCCESS;
}

extern int node_config_load(List gres_conf_list)
{
	int rc = SLURM_SUCCESS;
	ListIterator itr;
	gres_slurmd_conf_t *gres_slurmd_conf;
	char *identifier, *colon, *one_device;
	hostlist_t device_hl;
	gres_device_t *gres_device;

	if (gres_devices)
		return rc;

	xassert(gres_conf_list);

	itr = list_iterator_create(gres_conf_list);
	while ((gres_slurmd_conf = list_next(itr))) {
		if ((gres_slurmd_conf->has_file != 1) ||
		    !gres_slurmd_conf->file ||
		    !gres_slurmd_conf->has_identifier ||
		    !gres_slurmd_conf->identifier ||
		    xstrcmp(gres_slurmd_conf->name, gres_name))
			continue;

		identifier = xstrdup(gres_slurmd_conf->identifier);
		colon = strrchr(identifier, ':');
		if (colon) {
			device_hl = hostlist_create(colon + 1);
			colon[1] = '\0';
		} else {
			device_hl = hostlist_create(identifier);
			identifier[0] = '\0';
		}
		if (!device_hl) {
			error("can't parse gres.conf identifier record (%s)", gres_slurmd_conf->identifier);
			xfree(identifier);
			continue;
		}

		while ((one_device = hostlist_shift(device_hl))) {
			if (!gres_devices)
				gres_devices = list_create(destroy_gres_device);

			gres_device = xmalloc(sizeof(gres_device_t));
			list_append(gres_devices, gres_device);

			gres_device->path = xstrdup(gres_slurmd_conf->file);
			xstrfmtcat(gres_device->major, "%s%s",
			           identifier, one_device);
			gres_device->dev_num = (gres_device_num++);

			info("%s device number %d(%s):%s",
			    gres_name, gres_device->dev_num,
			    gres_device->path, gres_device->major);
			free(one_device);
		}
		hostlist_destroy(device_hl);
		xfree(identifier);
	}
	list_iterator_destroy(itr);

	return rc;
}

extern void job_set_env(char ***job_env_ptr, void *gres_ptr, int node_inx)
{
	bool already_seen = false;

	_set_env(job_env_ptr, gres_ptr, node_inx, NULL,
		 &already_seen, false, true);
}

extern void step_set_env(char ***step_env_ptr, void *gres_ptr)
{
	static bool already_seen = false;

	_set_env(step_env_ptr, gres_ptr, 0, NULL,
		 &already_seen, false, false);
}

extern void step_reset_env(char ***step_env_ptr, void *gres_ptr,
			   bitstr_t *usable_gres)
{
	static bool already_seen = false;

	_set_env(step_env_ptr, gres_ptr, 0, usable_gres,
		 &already_seen, true, false);
}

extern void send_stepd(int fd)
{
	common_send_stepd(fd, gres_devices);
}

extern void recv_stepd(int fd)
{
	common_recv_stepd(fd, &gres_devices);
}

extern int job_info(gres_job_state_t *job_gres_data, uint32_t node_inx,
		     enum gres_job_data_type data_type, void *data)
{
	return EINVAL;
}

extern int step_info(gres_step_state_t *step_gres_data, uint32_t node_inx,
		     enum gres_step_data_type data_type, void *data)
{
	return EINVAL;
}

extern List get_devices(void)
{
	return gres_devices;
}
