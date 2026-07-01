//==============================================================================
// arbiter_runner.js - Persistent independent arbiter (authoritative judge)
//
// Stateless per-query: maintains no state between messages. The orchestrator
// sends state + move, arbiter validates / simulates / checks terminal / etc.
//
// This is the Python-side "ground truth" that neither player controls.
// It uses the same epilog + general.js stack so semantics are identical,
// but runs in its own subprocess independent from any player.
//
// Protocol (JSON per line):
//   {"type":"load","game":"tictactoe"}                     -> {"ok":true,"result":"loaded"}
//   {"type":"init"}                                         -> {"ok":true,"result":[initial state]}
//   {"type":"legals","state":[...]}                         -> {"ok":true,"result":[[action],...]}
//   {"type":"legals_for","state":[...],"role":"x"}          -> {"ok":true,"result":[[action],...]}
//     (for HRF where legal has arity 1, this equals legals when control=role)
//   {"type":"is_legal","state":[...],"move":[...]}          -> {"ok":true,"result":true|false}
//   {"type":"simulate","state":[...],"move":[...]}          -> {"ok":true,"result":[new state]}
//   {"type":"terminal","state":[...]}                       -> {"ok":true,"result":true|false}
//   {"type":"reward","state":[...],"role":"x"}              -> {"ok":true,"result":100}
//   {"type":"control","state":[...]}                        -> {"ok":true,"result":"x"}
//   {"type":"exit"}                                         -> process exits
//==============================================================================

"use strict";

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const readline = require('readline');

global.indexing = false;
global.dataindexing = false;
global.ruleindexing = true;

// Load Epilog
vm.runInThisContext(
  fs.readFileSync(path.join(__dirname, 'epilog.js'), 'utf-8'),
  { filename: 'epilog.js' });

// Load general.js (GGP API)
try {
  vm.runInThisContext(
    fs.readFileSync(path.join(__dirname, 'general.js'), 'utf-8'),
    { filename: 'general.js' });
} catch (e) {
  // module.exports fails in script context, ignore
  if (!/module is not defined|exports is not defined/.test(String(e))) {
    process.stderr.write("general.js error: " + e + "\n");
  }
}

let currentLibrary = null;
let currentGame = null;

function safeCall(fn) {
  try { return { ok: true, result: fn() }; }
  catch (e) { return { ok: false, error: String(e && e.message || e),
                        stack: e && e.stack ? e.stack : null }; }
}

function respond(obj) {
  process.stdout.write(JSON.stringify(obj) + "\n");
}

function loadGame(gameName) {
  const rulesPath = path.join(__dirname, '..', 'downloads',
                              gameName + '_rulesheet.hrf');
  if (!fs.existsSync(rulesPath)) {
    throw new Error("Rules file not found: " + rulesPath);
  }
  const text = fs.readFileSync(rulesPath, 'utf-8');
  const parsed = global.readdata(text);
  currentLibrary = global.definemorerules([], parsed);
  currentGame = gameName;
  return { rules: parsed.length, library: currentLibrary.length };
}

const rl = readline.createInterface({ input: process.stdin, terminal: false });

rl.on('line', (line) => {
  line = line.trim();
  if (!line) return;

  let msg;
  try { msg = JSON.parse(line); }
  catch (e) { respond({ ok: false, error: "bad_json: " + e.message }); return; }

  if (msg.type === 'load') {
    try {
      const info = loadGame(msg.game);
      respond({ ok: true, result: 'loaded', info: info });
    } catch (e) {
      respond({ ok: false, error: String(e.message || e) });
    }
    return;
  }

  if (!currentLibrary) {
    respond({ ok: false, error: "no game loaded; send {type:'load',game:...} first" });
    return;
  }

  if (msg.type === 'init') {
    respond(safeCall(() => global.findinits(currentLibrary)));
  }
  else if (msg.type === 'legals') {
    respond(safeCall(() => global.findlegals(msg.state, currentLibrary)));
  }
  else if (msg.type === 'is_legal') {
    respond(safeCall(() => {
      const legals = global.findlegals(msg.state, currentLibrary);
      const mv = JSON.stringify(msg.move);
      return legals.some(l => JSON.stringify(l) === mv);
    }));
  }
  else if (msg.type === 'simulate') {
    respond(safeCall(() => global.simulate(msg.move, msg.state, currentLibrary)));
  }
  else if (msg.type === 'terminal') {
    respond(safeCall(() => global.findterminalp(msg.state, currentLibrary)));
  }
  else if (msg.type === 'reward') {
    respond(safeCall(() => {
      const v = global.findreward(msg.role, msg.state, currentLibrary);
      return typeof v === 'string' ? parseInt(v, 10) : v;
    }));
  }
  else if (msg.type === 'control') {
    respond(safeCall(() => global.findcontrol(msg.state, currentLibrary)));
  }
  else if (msg.type === 'roles') {
    respond(safeCall(() => global.findroles(currentLibrary)));
  }
  else if (msg.type === 'exit') {
    process.exit(0);
  }
  else {
    respond({ ok: false, error: "unknown_type: " + msg.type });
  }
});

rl.on('close', () => process.exit(0));

process.stdout.write(JSON.stringify({ ok: true, result: 'arbiter_loaded' }) + "\n");
