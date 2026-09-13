// One command build, because the platform runs `build` as an argv array and
// an argv array cannot contain a shell `&&`.
//
//   bun run aias:build  =  install + next build + assemble the standalone tree
//
// Next writes the standalone server to .next/standalone/server.js but leaves
// .next/static and public where they are, so they are copied here as the Next
// docs require. After this script the app starts with:
//   node .next/standalone/server.js
import { cpSync, existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { join } from "node:path";

const root = process.cwd();

function run(command, args) {
  console.log(`> ${command} ${args.join(" ")}`);
  const result = spawnSync(command, args, { stdio: "inherit", cwd: root });
  if (result.status !== 0) {
    console.error(`${command} ${args.join(" ")} failed with status ${result.status}`);
    process.exit(result.status ?? 1);
  }
}

run("bun", ["install"]);
run("bun", ["run", "build"]);

const standalone = join(root, ".next", "standalone");
if (!existsSync(join(standalone, "server.js"))) {
  console.error("next build did not produce .next/standalone/server.js, is output: 'standalone' set?");
  process.exit(1);
}

cpSync(join(root, ".next", "static"), join(standalone, ".next", "static"), { recursive: true });
if (existsSync(join(root, "public"))) {
  cpSync(join(root, "public"), join(standalone, "public"), { recursive: true });
}

console.log("build ready: node .next/standalone/server.js");
