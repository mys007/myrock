package = "myrock"
version = "1.0-0"

source = {
   url = "...",
   tag = "master"
}

description = {
   summary = "MyRock",
   detailed = [[
Universite Paris-Est MLV Imagine laboratory nn routines
   ]],
      license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "nn",
   "cunn"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"
   ]],
   install_command = "cd build && $(MAKE) install"
}
