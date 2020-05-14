""" TensorFlow Http Archive

Modified http_archive that allows us to override the TensorFlow commit that is
downloaded by setting an environment variable. This override is to be used for
testing purposes.

Add the following to your Bazel build command in order to override the
TensorFlow revision.

build: --action_env TF_REVISION="<git commit hash>"

  * `TF_REVISION`: tensorflow revision override (git commit hash)
"""

_TF_REVISION = "TF_REVISION"

def _get_env_var(ctx, name):
    if name in ctx.os.environ:
        return ctx.os.environ[name]
    else:
        return None

# Checks if we should use the system lib instead of the bundled one
def _use_system_lib(ctx, name):
    syslibenv = _get_env_var(ctx, "TF_SYSTEM_LIBS")
    if syslibenv:
        for n in syslibenv.strip().split(","):
            if n.strip() == name:
                return True
    return False

# Executes specified command with arguments and calls 'fail' if it exited with
# non-zero code
def _execute_and_check_ret_code(repo_ctx, cmd_and_args):
    result = repo_ctx.execute(cmd_and_args, timeout = 60)
    if result.return_code != 0:
        fail(("Non-zero return code({1}) when executing '{0}':\n" + "Stdout: {2}\n" +
              "Stderr: {3}").format(
            " ".join(cmd_and_args),
            result.return_code,
            result.stdout,
            result.stderr,
        ))

def _repos_are_siblings():
    return Label("@foo//bar").workspace_root.startswith("../")

# Apply a patch_file to the repository root directory
# Runs 'patch -p1'
def _apply_patch(ctx, patch_file):
    if not ctx.which("patch"):
        fail("patch command is not found, please install it")
    cmd = ["patch", "-p1", "-d", ctx.path("."), "-i", ctx.path(patch_file)]
    _execute_and_check_ret_code(ctx, cmd)

def _tensorflow_http_archive(ctx):
    use_syslib = _use_system_lib(ctx, ctx.attr.name)

    # Work around the bazel bug that redownloads the whole library.
    # Remove this after https://github.com/bazelbuild/bazel/issues/10515 is fixed.
    if ctx.attr.additional_build_files:
        for internal_src in ctx.attr.additional_build_files:
            _ = ctx.path(Label(internal_src))

    # End of workaround.

    if not use_syslib:
        ctx.download_and_extract(
            ctx.attr.urls,
            "",
            ctx.attr.sha256,
            ctx.attr.type,
            ctx.attr.strip_prefix,
        )
        if ctx.attr.patch_file != None:
            _apply_patch(ctx, ctx.attr.patch_file)

    if use_syslib and ctx.attr.system_build_file != None:
        # Use BUILD.bazel to avoid conflict with third party projects with
        # BUILD or build (directory) underneath.
        ctx.template("BUILD.bazel", ctx.attr.system_build_file, {
            "%prefix%": ".." if _repos_are_siblings() else "external",
        }, False)

    elif ctx.attr.build_file != None:
        # Use BUILD.bazel to avoid conflict with third party projects with
        # BUILD or build (directory) underneath.
        ctx.template("BUILD.bazel", ctx.attr.build_file, {
            "%prefix%": ".." if _repos_are_siblings() else "external",
        }, False)

    if use_syslib:
        for internal_src, external_dest in ctx.attr.system_link_files.items():
            ctx.symlink(Label(internal_src), ctx.path(external_dest))

    if ctx.attr.additional_build_files:
        for internal_src, external_dest in ctx.attr.additional_build_files.items():
            ctx.symlink(Label(internal_src), ctx.path(external_dest))
    

tensorflow_http_archive = repository_rule(
    attrs = {
        "sha256": attr.string(mandatory = True),
        "urls": attr.string_list(
            mandatory = True,
            allow_empty = False,
        ),
        "strip_prefix": attr.string(),
        "type": attr.string(),
        "patch_file": attr.label(),
        "build_file": attr.label(),
        "system_build_file": attr.label(),
        "system_link_files": attr.string_dict(),
        "additional_build_files": attr.string_dict(),
    },
    environ = [
        "TF_SYSTEM_LIBS",
    ],
    implementation = _tensorflow_http_archive,
)
