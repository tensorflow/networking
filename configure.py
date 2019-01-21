# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""configure script to get build parameters from user."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import errno
import os
import platform
import re
import subprocess
import sys

# pylint: disable=g-import-not-at-top
try:
  from shutil import which
except ImportError:
  from distutils.spawn import find_executable as which
# pylint: enable=g-import-not-at-top


_DEFAULT_PROMPT_ASK_ATTEMPTS = 10

_TF_BAZELRC_FILENAME = '.tf_networking_configure.bazelrc'
_TF_WORKSPACE_ROOT = ''
_TF_BAZELRC = ''


class UserInputError(Exception):
  pass


def is_windows():
  return platform.system() == 'Windows'


def is_linux():
  return platform.system() == 'Linux'


def is_macos():
  return platform.system() == 'Darwin'


def is_ppc64le():
  return platform.machine() == 'ppc64le'


def is_cygwin():
  return platform.system().startswith('CYGWIN_NT')


def get_input(question):
  try:
    try:
      answer = raw_input(question)
    except NameError:
      answer = input(question)  # pylint: disable=bad-builtin
  except EOFError:
    answer = ''
  return answer


def symlink_force(target, link_name):
  """Force symlink, equivalent of 'ln -sf'.

  Args:
    target: items to link to.
    link_name: name of the link.
  """
  try:
    os.symlink(target, link_name)
  except OSError as e:
    if e.errno == errno.EEXIST:
      os.remove(link_name)
      os.symlink(target, link_name)
    else:
      raise e


def write_to_bazelrc(line):
  with open(_TF_BAZELRC, 'a') as f:
    f.write(line + '\n')


def write_action_env_to_bazelrc(var_name, var):
  write_to_bazelrc('build --action_env %s="%s"' % (var_name, str(var)))


def run_shell(cmd, allow_non_zero=False):
  if allow_non_zero:
    try:
      output = subprocess.check_output(cmd)
    except subprocess.CalledProcessError as e:
      output = e.output
  else:
    output = subprocess.check_output(cmd)
  return output.decode('UTF-8').strip()


def cygpath(path):
  """Convert path from posix to windows."""
  return os.path.abspath(path).replace('\\', '/')


def reset_tf_configure_bazelrc():
  """Reset file that contains customized config settings."""
  open(_TF_BAZELRC, 'w').close()


def get_var(environ_cp,
            var_name,
            query_item,
            enabled_by_default,
            question=None,
            yes_reply=None,
            no_reply=None):
  """Get boolean input from user.

  If var_name is not set in env, ask user to enable query_item or not. If the
  response is empty, use the default.

  Args:
    environ_cp: copy of the os.environ.
    var_name: string for name of environment variable, e.g. "TF_NEED_HDFS".
    query_item: string for feature related to the variable, e.g. "Hadoop File
      System".
    enabled_by_default: boolean for default behavior.
    question: optional string for how to ask for user input.
    yes_reply: optional string for reply when feature is enabled.
    no_reply: optional string for reply when feature is disabled.

  Returns:
    boolean value of the variable.

  Raises:
    UserInputError: if an environment variable is set, but it cannot be
      interpreted as a boolean indicator, assume that the user has made a
      scripting error, and will continue to provide invalid input.
      Raise the error to avoid infinitely looping.
  """
  if not question:
    question = 'Do you wish to build TensorFlow with %s support?' % query_item
  if not yes_reply:
    yes_reply = '%s support will be enabled for TensorFlow.' % query_item
  if not no_reply:
    no_reply = 'No %s' % yes_reply

  yes_reply += '\n'
  no_reply += '\n'

  if enabled_by_default:
    question += ' [Y/n]: '
  else:
    question += ' [y/N]: '

  var = environ_cp.get(var_name)
  if var is not None:
    var_content = var.strip().lower()
    true_strings = ('1', 't', 'true', 'y', 'yes')
    false_strings = ('0', 'f', 'false', 'n', 'no')
    if var_content in true_strings:
      var = True
    elif var_content in false_strings:
      var = False
    else:
      raise UserInputError(
          'Environment variable %s must be set as a boolean indicator.\n'
          'The following are accepted as TRUE : %s.\n'
          'The following are accepted as FALSE: %s.\n'
          'Current value is %s.' % (var_name, ', '.join(true_strings),
                                    ', '.join(false_strings), var))

  while var is None:
    user_input_origin = get_input(question)
    user_input = user_input_origin.strip().lower()
    if user_input == 'y':
      print(yes_reply)
      var = True
    elif user_input == 'n':
      print(no_reply)
      var = False
    elif not user_input:
      if enabled_by_default:
        print(yes_reply)
        var = True
      else:
        print(no_reply)
        var = False
    else:
      print('Invalid selection: %s' % user_input_origin)
  return var


def set_build_var(environ_cp,
                  var_name,
                  query_item,
                  option_name,
                  enabled_by_default,
                  bazel_config_name=None):
  """Set if query_item will be enabled for the build.

  Ask user if query_item will be enabled. Default is used if no input is given.
  Set subprocess environment variable and write to .bazelrc if enabled.

  Args:
    environ_cp: copy of the os.environ.
    var_name: string for name of environment variable, e.g. "TF_NEED_HDFS".
    query_item: string for feature related to the variable, e.g. "Hadoop File
      System".
    option_name: string for option to define in .bazelrc.
    enabled_by_default: boolean for default behavior.
    bazel_config_name: Name for Bazel --config argument to enable build feature.
  """

  var = str(int(get_var(environ_cp, var_name, query_item, enabled_by_default)))
  environ_cp[var_name] = var
  # if var == '1':
  #   write_to_bazelrc(
  #       'build:%s --define %s=true' % (bazel_config_name, option_name))
  #   write_to_bazelrc('build --config=%s' % bazel_config_name)
  # elif bazel_config_name is not None:
  #   # TODO(mikecase): Migrate all users of configure.py to use --config Bazel
  #   # options and not to set build configs through environment variables.
  #   write_to_bazelrc(
  #       'build:%s --define %s=true' % (bazel_config_name, option_name))


def set_action_env_var(environ_cp,
                       var_name,
                       query_item,
                       enabled_by_default,
                       question=None,
                       yes_reply=None,
                       no_reply=None):
  """Set boolean action_env variable.

  Ask user if query_item will be enabled. Default is used if no input is given.
  Set environment variable and write to .bazelrc.

  Args:
    environ_cp: copy of the os.environ.
    var_name: string for name of environment variable, e.g. "TF_NEED_HDFS".
    query_item: string for feature related to the variable, e.g. "Hadoop File
      System".
    enabled_by_default: boolean for default behavior.
    question: optional string for how to ask for user input.
    yes_reply: optional string for reply when feature is enabled.
    no_reply: optional string for reply when feature is disabled.
  """
  var = int(
      get_var(environ_cp, var_name, query_item, enabled_by_default, question,
              yes_reply, no_reply))

  write_action_env_to_bazelrc(var_name, var)
  environ_cp[var_name] = str(var)


def get_from_env_or_user_or_default(environ_cp, var_name, ask_for_var,
                                    var_default):
  """Get var_name either from env, or user or default.

  If var_name has been set as environment variable, use the preset value, else
  ask for user input. If no input is provided, the default is used.

  Args:
    environ_cp: copy of the os.environ.
    var_name: string for name of environment variable, e.g. "TF_NEED_HDFS".
    ask_for_var: string for how to ask for user input.
    var_default: default value string.

  Returns:
    string value for var_name
  """
  var = environ_cp.get(var_name)
  if not var:
    var = get_input(ask_for_var)
    print('\n')
  if not var:
    var = var_default
  return var


def prompt_loop_or_load_from_env(environ_cp,
                                 var_name,
                                 var_default,
                                 ask_for_var,
                                 check_success,
                                 error_msg,
                                 suppress_default_error=False,
                                 n_ask_attempts=_DEFAULT_PROMPT_ASK_ATTEMPTS):
  """Loop over user prompts for an ENV param until receiving a valid response.

  For the env param var_name, read from the environment or verify user input
  until receiving valid input. When done, set var_name in the environ_cp to its
  new value.

  Args:
    environ_cp: (Dict) copy of the os.environ.
    var_name: (String) string for name of environment variable, e.g. "TF_MYVAR".
    var_default: (String) default value string.
    ask_for_var: (String) string for how to ask for user input.
    check_success: (Function) function that takes one argument and returns a
      boolean. Should return True if the value provided is considered valid. May
      contain a complex error message if error_msg does not provide enough
      information. In that case, set suppress_default_error to True.
    error_msg: (String) String with one and only one '%s'. Formatted with each
      invalid response upon check_success(input) failure.
    suppress_default_error: (Bool) Suppress the above error message in favor of
      one from the check_success function.
    n_ask_attempts: (Integer) Number of times to query for valid input before
      raising an error and quitting.

  Returns:
    [String] The value of var_name after querying for input.

  Raises:
    UserInputError: if a query has been attempted n_ask_attempts times without
      success, assume that the user has made a scripting error, and will
      continue to provide invalid input. Raise the error to avoid infinitely
      looping.
  """
  default = environ_cp.get(var_name) or var_default
  full_query = '%s [Default is %s]: ' % (
      ask_for_var,
      default,
  )

  for _ in range(n_ask_attempts):
    val = get_from_env_or_user_or_default(environ_cp, var_name, full_query,
                                          default)
    if check_success(val):
      break
    if not suppress_default_error:
      print(error_msg % val)
    environ_cp[var_name] = ''
  else:
    raise UserInputError(
        'Invalid %s setting was provided %d times in a row. '
        'Assuming to be a scripting mistake.' % (var_name, n_ask_attempts))

  environ_cp[var_name] = val
  return val


def set_mpi_home(environ_cp):
  """Set MPI_HOME."""

  default_mpi_home = which('mpirun') or which('mpiexec') or ''
  default_mpi_home = os.path.dirname(os.path.dirname(default_mpi_home))

  def valid_mpi_path(mpi_home):
    exists = (
        os.path.exists(os.path.join(mpi_home, 'include')) and
        (os.path.exists(os.path.join(mpi_home, 'lib')) or
         os.path.exists(os.path.join(mpi_home, 'lib64')) or
         os.path.exists(os.path.join(mpi_home, 'lib32'))))
    if not exists:
      print(
          'Invalid path to the MPI Toolkit. %s or %s or %s or %s cannot be found'
          % (os.path.join(mpi_home, 'include'),
             os.path.exists(os.path.join(mpi_home, 'lib')),
             os.path.exists(os.path.join(mpi_home, 'lib64')),
             os.path.exists(os.path.join(mpi_home, 'lib32'))))
    return exists

  _ = prompt_loop_or_load_from_env(
      environ_cp,
      var_name='MPI_HOME',
      var_default=default_mpi_home,
      ask_for_var='Please specify the MPI toolkit folder.',
      check_success=valid_mpi_path,
      error_msg='',
      suppress_default_error=True)


def set_other_mpi_vars(environ_cp):
  """Set other MPI related variables."""
  # Link the MPI header files
  mpi_home = environ_cp.get('MPI_HOME')

  # Determine the location of the MPI header files
  include_home = ""
  if os.path.exists(os.path.join(mpi_home, 'include/mpi.h')):
    symlink_force('%s/include/mpi.h' % mpi_home, 'third_party/mpi/mpi.h')
    include_home = mpi_home + "/include"
  elif os.path.exists(os.path.join(mpi_home, 'include/mpi/mpi.h')):
    symlink_force('%s/include/mpi/mpi.h' % mpi_home, 'third_party/mpi/mpi.h')
    include_home = mpi_home + "/include/mpi"
  else:
    raise ValueError(
        'Cannot find the MPI header file in %s/include or %s/include/mpi' %
        mpi_home, mpi_home)

  # Determine if we use OpenMPI or MVAPICH, these require different header files
  # to be included here to make bazel dependency checker happy
  if os.path.exists(os.path.join(include_home, 'mpi_portable_platform.h')):
    symlink_force(
        os.path.join(include_home, 'mpi_portable_platform.h'),
        'third_party/mpi/mpi_portable_platform.h')
    write_to_bazelrc("build --define mpi_library_is_openmpi_based=true")
  else:
    # MVAPICH / MPICH
    symlink_force(
        os.path.join(include_home, 'mpio.h'), 'third_party/mpi/mpio.h')
    symlink_force(
        os.path.join(include_home, 'mpicxx.h'), 'third_party/mpi/mpicxx.h')
    write_to_bazelrc("build --define mpi_library_is_openmpi_based=false")

  if os.path.exists(os.path.join(mpi_home, 'lib/libmpi.so')):
    symlink_force(
        os.path.join(mpi_home, 'lib/libmpi.so'), 'third_party/mpi/libmpi.so')
  elif os.path.exists(os.path.join(mpi_home, 'lib64/libmpi.so')):
    symlink_force(
        os.path.join(mpi_home, 'lib64/libmpi.so'), 'third_party/mpi/libmpi.so')
  elif os.path.exists(os.path.join(mpi_home, 'lib32/libmpi.so')):
    symlink_force(
        os.path.join(mpi_home, 'lib32/libmpi.so'), 'third_party/mpi/libmpi.so')
  else:
    raise ValueError(
        'Cannot find the MPI library file in %s/lib or %s/lib64 or %s/lib32' %
        mpi_home, mpi_home, mpi_home)


def config_info_line(name, help_text):
  """Helper function to print formatted help text for Bazel config options."""
  print('\t--config=%-12s\t# %s' % (name, help_text))


def main():
  global _TF_WORKSPACE_ROOT
  global _TF_BAZELRC

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--workspace',
      type=str,
      default=os.path.abspath(os.path.dirname(__file__)),
      help='The absolute path to your active Bazel workspace.')
  args = parser.parse_args()

  _TF_WORKSPACE_ROOT = args.workspace
  _TF_BAZELRC = os.path.join(_TF_WORKSPACE_ROOT, _TF_BAZELRC_FILENAME)

  # Make a copy of os.environ to be clear when functions and getting and setting
  # environment variables.
  environ_cp = dict(os.environ)

  reset_tf_configure_bazelrc()

  set_build_var(environ_cp, 'TF_NEED_MPI', 'MPI', 'with_mpi_support', False)
  if environ_cp.get('TF_NEED_MPI') == '1':
    set_mpi_home(environ_cp)
    set_other_mpi_vars(environ_cp)

  #config_info_line('gdr', 'Build with GDR support.')
  #config_info_line('verbs', 'Build with libverbs support.')


if __name__ == '__main__':
  main()
