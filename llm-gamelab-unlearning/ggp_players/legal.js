//==============================================================================
// legal.js
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
  return findlegalx(state,library)}

function stop (move)
 {return false}

function abort ()
 {return false}

//==============================================================================
// End of player code
//==============================================================================
