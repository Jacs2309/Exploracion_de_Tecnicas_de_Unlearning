//==============================================================================
// maximax.js
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
  return playmaximax(role)}

function stop (move)
 {return false}

function abort ()
 {return false}

//==============================================================================
// maximax//==============================================================================

var nodes = 0;
var terminals = 0;
var elapsed = 0;

function playmaximax (role) {var actions = shuffle(findlegals(state,library));
  if (actions.length===0) {return false};
  if (actions.length===1) {return actions[0]};
  var action = actions[0];
  var score = 0;
  nodes = 0
  for (var i=0; i<actions.length; i++)
      {console.log(grind(actions[i]));
       var newstate = simulate(actions[i],state,library);
       var newvector = maximax(newstate); console.log(newvector);
       if (newvector[role]===100) {return actions[i]};
       if (newvector[role]>score) {score = newvector[role]}};
  return action}

function testmaximax (role,state)
 {nodes = 0;
  terminals = 0;
  var beg = performance.now();
  var vector = maximax(state);
  var end = performance.now();
  elapsed = Math.round(end-beg);
  return vector[role]}

function maximax (state) {console.log(state);nodes = nodes + 1;
  if (findterminalp(state,library))
     {var vector = {};
      for (var i=0; i<roles.length; i++)
          {vector[roles[i]]=findreward(roles[i],state,library)*1};
      return vector};
  var active = findcontrol(state,library);
  var actions = findlegals(state,library);
  var vector = {};
  for (var i=0; i<roles.length; i++) {vector[roles[i]]=-1};
  for (var i=0; i<actions.length; i++)
      {var newstate = simulate(actions[i],state,library);
       var newvector = maximax(newstate);
       if (newvector[active]<vector[active]) {continue};
       if (newvector[active]>vector[active]) {vector = newvector};
       for (var j=0; j<roles.length; j++)
           {if (roles[j]!==active)
               {vector[roles[j]] = Math.min(vector[roles[j]],newvector[roles[j]])}}};
  return vector}

function shuffle (array)
 {for (var i = array.length-1; i>0; i--)
      {var j = Math.floor(Math.random() * (i + 1));
       var temp = array[i];
       array[i] = array[j];
       array[j] = temp};
  return array}

//==============================================================================
// End of player code
//==============================================================================
