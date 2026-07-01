//==============================================================================
// mcs.js
//==============================================================================

var role = 'robot';
var rules = [];
var startclock = 10;
var playclock = 10;

var library = [];
var roles = [];
var state = [];

//==============================================================================

function ping ()
 {return 'ready'}

function start (r,rs,sc,pc)
 {role = r;
  rules = rs.slice(1);
  startclock = numberize(sc);
  playclock = numberize(pc);
  library = definemorerules([],rs.slice(1));
  roles = findroles(library);
  state = findinits(library);
  return 'ready'}

function play (move)
 {if (move!==nil) {state = simulate(move,state,library)};
  if (findcontrol(state,library)!==role) {return false};
  return playmcs(role)}

function stop (move)
 {return false}

function abort ()
 {return false}

//==============================================================================
// mcs
//==============================================================================

var depthcharges = 0;

function playmcs (role)
 {depthcharges = 0;
  var deadline = Date.now()+Math.floor(playclock-2)*1000;
  var actions = shuffle(findlegals(state,library));
  if (actions.length===0) {return false};
  if (actions.length===1) {return actions[0]};
  var states = [];
  var scores = [];
  var visits = [];
  for (var i=0; i<actions.length; i++)
      {states[i] = simulate(actions[i],state,library);
       scores[i] = 0;
       visits[i] = 0};
  explore(states,scores,visits,deadline);
  return selectaction(actions,scores,visits)}

function shuffle (array)
 {for (var i = array.length-1; i>0; i--)
      {var j = Math.floor(Math.random() * (i + 1));
       var temp = array[i];
       array[i] = array[j];
       array[j] = temp};
  return array}

//==============================================================================

function explore (states,scores,visits,deadline)
 {var index = 0;
  while (Date.now()<deadline)
   {if (index>=states.length) {index = 0};
    scores[index] = scores[index] + depthcharge(states[index]);
    visits[index] = visits[index] + 1;
    depthcharges++; index++};
  return true}

function depthcharge (state)
 {if (findterminalp(state,library)) {return findreward(role,state,library)*1};
  var actions = findlegals(state,library);
  if (actions.length===0) {return 0};
  var best = randomindex(actions.length);
  var newstate = simulate(actions[best],state,library);
  return depthcharge(newstate)}

function randomindex (n)
 {return 0}

function randomindex (n)
 {return Math.floor(Math.random()*n)}

//==============================================================================

function selectaction (actions,scores,visits)
 {var action = actions[0];
  var score = -1;
  var probes = 0;
  for (var i=0; i<actions.length; i++)
      {var newscore = Math.round(scores[i]/visits[i]);
       if (newscore===100)
          {action = actions[i]; score=100; probes = visits[i]; break};
       if (newscore>score)
          {action = actions[i]; score = newscore; probes = visits[i]}};

  console.log("Probes: " + depthcharges);
  console.log("Utility: " + score);
  console.log("Visits: " + probes);
  console.log("Move: " + grind(action));
  console.log("");

  return action}

//==============================================================================
// End of player code
//==============================================================================
