//==============================================================================
// ggp_player_runner.js - Persistent subprocess for a GGP player
//
// Usage: node ggp_player_runner.js <player_name> <game_name>
//   player_name: random | legal | minimax | mcs | greedy | maximax
//   game_name:   tictactoe | connectfour | suicide | ... (looks up downloads/<game>_rulesheet.hrf)
//
// Loads (in order):
//   1. epilog.js  - Stanford logic programming interpreter
//   2. general.js - GGP API (findroles, findinits, findlegals, simulate, ...)
//   3. <player>.js - The specific player (defines ping/start/play/stop/abort)
//
// Protocol (one JSON object per line on stdin; one JSON per line on stdout):
//   {"type":"ping"}                             -> {"ok":true, "result":"ready"}
//   {"type":"start", "role":"x",
//    "startclock":10, "playclock":10}           -> {"ok":true, "result":"ready"}
//   {"type":"play",  "move": [...] | "nil"}     -> {"ok":true, "result":[...move...]}
//   {"type":"stop",  "move": [...]}             -> {"ok":true, "result":false}
//   {"type":"abort"}                            -> {"ok":true, "result":false}
//   {"type":"state"}                            -> {"ok":true, "result":[...current state...]}
//   {"type":"legals"}                           -> {"ok":true, "result":[[...],[...],...]}
//   {"type":"terminal"}                         -> {"ok":true, "result":true|false}
//   {"type":"reward","role":"x"}                -> {"ok":true, "result":100}
//   {"type":"exit"}                             -> process exits
//
// The player's top-level variables (role, state, library, ...) persist between
// calls, so `play()` works incrementally just like in Stanford's GGP architecture.
//==============================================================================

"use strict";

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const readline = require('readline');

// CRITICAL: Redirect console.log/warn/info from players to stderr.
// Player scripts (mcs, greedy, maximax) print debug info with console.log.
// Those writes would corrupt our stdout JSON RPC channel. We override the
// console methods BEFORE loading any player code.
const origConsole = console;
const stderrConsole = new console.Console({ stdout: process.stderr, stderr: process.stderr });
global.console = stderrConsole;

const args = process.argv.slice(2);
if (args.length < 2) {
  process.stderr.write("Usage: node ggp_player_runner.js <player_name> <game_name> [seed]\n");
  process.exit(1);
}
const [playerName, gameName] = args;
const seed = args[2] ? parseInt(args[2], 10) : null;

// Seedeable PRNG (mulberry32) — replaces Math.random when a seed is provided.
// This ensures reproducible experiments. Without a seed, V8's default is used.
if (seed !== null && !isNaN(seed)) {
  let s = seed | 0;
  Math.random = function() {
    s = (s + 0x6D2B79F5) | 0;
    var t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  process.stderr.write("PRNG seeded: " + seed + "\n");
}

// Epilog needs these config vars BEFORE its code runs (it reads them at top level)
global.indexing = false;
global.dataindexing = false;
global.ruleindexing = true;

// ---- Load Epilog into the global scope -------------------------------------
// Use runInThisContext so top-level `function foo(){}` attaches to global.
const epilogPath = path.resolve(__dirname, 'epilog.js');
const epilogCode = fs.readFileSync(epilogPath, 'utf-8');
vm.runInThisContext(epilogCode, { filename: 'epilog.js' });

// ---- Load the GGP API layer (general.js) -----------------------------------
const generalPath = path.resolve(__dirname, 'general.js');
const generalCode = fs.readFileSync(generalPath, 'utf-8');
// general.js does `module.exports = {...}` at the end. Running it with
// runInThisContext throws because `module` is defined but the function decls
// still attach to global. That's what we want.
try {
  vm.runInThisContext(generalCode, { filename: 'general.js' });
} catch (e) {
  // The `module.exports` line fails because we are not in a CommonJS wrapper.
  // That's fine — the function declarations already attached to global
  // before the error. Ignore the error if it's about module/exports.
  if (!/module is not defined|exports is not defined/.test(String(e))) {
    process.stderr.write("Error loading general.js: " + e + "\n");
  }
}

// ---- Load the player script -------------------------------------------------
// IMPORTANT: Some player scripts (greedy.js) define a function named `process`
// which would shadow Node's global `process` object via vm.runInThisContext.
// Same risk for other critical globals. We save a reference to the real process
// BEFORE loading the player, and restore it AFTER — along with any other globals
// the player might inadvertently clobber.
const playerPath = path.resolve(__dirname, playerName + '.js');
if (!fs.existsSync(playerPath)) {
  process.stderr.write("Player file not found: " + playerPath + "\n");
  process.exit(1);
}
const savedGlobals = {
  process: global.process,
  require: global.require,
  console: global.console,
  setTimeout: global.setTimeout,
  setImmediate: global.setImmediate,
  Buffer: global.Buffer,
};
const playerCode = fs.readFileSync(playerPath, 'utf-8');
vm.runInThisContext(playerCode, { filename: playerName + '.js' });

// Expose player's top-level functions (ping/start/play/stop/abort) under
// wrappers, then restore Node globals that the player may have shadowed.
const playerPing  = global.ping;
const playerStart = global.start;
const playerPlay  = global.play;
const playerStop  = global.stop;
const playerAbort = global.abort;

for (const k of Object.keys(savedGlobals)) {
  global[k] = savedGlobals[k];
}
// process is now the real Node process again.

// ---- Load the game rules (KIF text) ----------------------------------------
const rulesPath = path.resolve(__dirname, '..', 'downloads', gameName + '_rulesheet.hrf');
if (!fs.existsSync(rulesPath)) {
  process.stderr.write("Rules file not found: " + rulesPath + "\n");
  process.stderr.write("Expected at: downloads/" + gameName + "_rulesheet.hrf\n");
  process.exit(1);
}
const rulesText = fs.readFileSync(rulesPath, 'utf-8');

// Parse KIF text into Epilog's internal AST using readdata.
// Stanford's convention: rs.slice(1) is done inside start(), so we need to
// pass a list whose first element is a dummy header (the parsed rules list
// itself after readdata starts with a header-less sequence, so we prepend one).
const parsedRules = global.readdata(rulesText);

// Stanford's players do `rules = rs.slice(1)` inside start(), assuming the
// first element is some header (GGP message type). We match that convention.
const rulesForStart = ['metagame'].concat(parsedRules);

// ---- RPC server --------------------------------------------------------------

function safeCall(fn, ...a) {
  try {
    return { ok: true, result: fn.apply(null, a) };
  } catch (e) {
    return { ok: false, error: String(e && e.message || e),
             stack: e && e.stack ? e.stack : null };
  }
}

function respond(obj) {
  try {
    process.stdout.write(JSON.stringify(obj) + "\n");
  } catch (e) {
    process.stdout.write(JSON.stringify({
      ok: false, error: "serialization_failed: " + String(e) }) + "\n");
  }
}

function normalizeMove(m) {
  if (m === null || m === undefined || m === 'nil' || m === 'NIL') {
    return global.nil;
  }
  return m;
}

const rl = readline.createInterface({ input: process.stdin, terminal: false });

rl.on('line', (line) => {
  line = line.trim();
  if (!line) return;

  let msg;
  try { msg = JSON.parse(line); }
  catch (e) { respond({ ok: false, error: "bad_json: " + e.message }); return; }

  const type = msg.type;

  if (type === 'ping') {
    respond(safeCall(playerPing));
  }
  else if (type === 'start') {
    const role = msg.role;
    const sc = msg.startclock !== undefined ? msg.startclock : 10;
    const pc = msg.playclock !== undefined ? msg.playclock : 10;
    respond(safeCall(playerStart, role, rulesForStart, sc, pc));
  }
  else if (type === 'play') {
    const move = normalizeMove(msg.move);
    respond(safeCall(playerPlay, move));
  }
  else if (type === 'stop') {
    const move = normalizeMove(msg.move);
    respond(safeCall(playerStop, move));
  }
  else if (type === 'abort') {
    respond(safeCall(playerAbort));
  }
  else if (type === 'state') {
    respond({ ok: true, result: global.state });
  }
  else if (type === 'legals') {
    respond(safeCall(global.findlegals, global.state, global.library));
  }
  else if (type === 'terminal') {
    respond(safeCall(global.findterminalp, global.state, global.library));
  }
  else if (type === 'reward') {
    const role = msg.role || global.role;
    respond(safeCall(global.findreward, role, global.state, global.library));
  }
  else if (type === 'control') {
    respond(safeCall(global.findcontrol, global.state, global.library));
  }
  else if (type === 'exit') {
    process.exit(0);
  }
  else {
    respond({ ok: false, error: "unknown_type: " + type });
  }
});

rl.on('close', () => process.exit(0));

// Notify parent we're ready. Parent should wait for this before sending ping.
process.stdout.write(JSON.stringify({ ok: true, result: "runner_loaded",
  player: playerName, game: gameName }) + "\n");
