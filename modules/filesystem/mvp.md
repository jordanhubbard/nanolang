# filesystem module MVP

<!--nl-snippet {"name":"module_filesystem_mvp","check":false}-->
```nano
from "modules/filesystem/filesystem.nano" import nl_fs_parent_dir, nl_fs_file_exists

fn main() -> int {
    let mut parent: string = ""
    let mut exists: int = 0
    unsafe { set parent (nl_fs_parent_dir "/tmp/demo.txt") }
    unsafe { set exists (nl_fs_file_exists "/tmp") }
    assert (>= (str_length parent) 0)
    assert (>= exists 0)
    return 0
}

shadow main { assert true }
```
