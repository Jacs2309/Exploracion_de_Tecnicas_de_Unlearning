//==============================================================================
// general.js
//==============================================================================
//==============================================================================
// Initialization//==============================================================================

indexing = false;
dataindexing = false;
ruleindexing = true;

//==============================================================================
// Basics
//==============================================================================

function findroles (rules)
 {return compfinds('R',seq('role','R'),seq(),rules)}

function findbases (rules)
 {return compfinds('P',seq('base','P'),seq(),rules)}

function findactions (rules)
 {return compfinds('A',seq('action','A'),seq(),rules)}

function findinits (rules)
 {return compfinds('P',seq('init','P'),seq(),rules)}

function findcontrol (facts,rules)
 {return compfindx('X',seq('control','X'),facts,rules)}

function findlegalp (move,facts,rules)
 {return compfindp(seq('legal',move),facts,rules)}

function findlegalx (facts,rules)
 {return compfindx('X',seq('legal','X'),facts,rules)}

function findlegals (facts,rules)
 {return compfinds('X',seq('legal','X'),facts,rules)}

function findreward (role,facts,rules)
 {var value = compfindx('R',seq('goal',role,'R'),facts,rules);
  if (value) {return value};
  return "0"}

function findterminalp (facts,rules)
 {return compfindp('terminal',facts,rules)}

//------------------------------------------------------------------------------

function simulate (move,state,rules)
 {var deltas = compexpand(move,state,rules);
  var additions = [];
  var deletions = [];
  for (var i=0; i<deltas.length; i++)
      {if (symbolp(deltas[i])) {additions.push(deltas[i]); continue};
       if (deltas[i][0]==='not') {deletions.push(deltas[i][1]); continue};
       additions.push(deltas[i])};
  var newstate = [];
  for (i = 0; i<state.length; i++)
      {if (find(state[i],additions)) {continue};
       if (find(state[i],deletions)) {continue};
       newstate.push(state[i])};
  return newstate.concat(additions)}
  
  // Exportar todas las funciones necesarias
module.exports = {
findroles,
findbases,
findactions,
findinits,
findcontrol,
findlegalp,
findlegalx,
findlegals,
findreward,
findterminalp,
simulate

};

//==============================================================================
//==============================================================================
//==============================================================================
