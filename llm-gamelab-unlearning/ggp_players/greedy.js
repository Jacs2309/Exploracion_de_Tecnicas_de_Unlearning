//==============================================================================
// greedy.js
//==============================================================================

function ping ()
 {return 'ready'}

function start (r,rs,sc,pc)
 {role = r;
  library = definemorerules([],rs.slice(1));
  roles = findroles(library);
  state = findinits(library);
  startclock = numberize(sc);
  playclock = numberize(pc);
  var active = findcontrol(state,library);
  var reward = parseInt(findreward(role,state,library));
  tree = makenode(state,active,reward,false);
  return 'ready'}

function play (move)
 {return playgreedy(move)}

function stop (move)
 {return false}

function abort ()
 {return false}

//==============================================================================

function makenode (state,mover,utility,complete)
 {return {state:state,
          actions:[],
          children:[],
          mover:mover,
          utility:utility,
          visits:0,
          complete:complete}}

//==============================================================================
// greedy
//==============================================================================

var tree = {};
var nodes = 0;
var terminals = 0;

function playgreedy (move)
 {if (move!==nil) {tree = subtree(move,tree); state = tree.state};

  nodes = 1;
  terminals = 0;

  if (findcontrol(state,library)!==role) {return false};
  var deadline = Date.now()+(playclock-2)*1000;
  while (Date.now()<deadline && !tree.complete) {processnode(tree)};
  move = selectaction(tree);

  console.log("Nodes: " + nodes);
  console.log("Terminals: " + terminals);
  console.log("Utility: " + tree.utility);
  console.log("");

  return move}

//==============================================================================

function subtree (move,node)
 {if (node.children.length===0)
     {var newstate = simulate(move,node.state,library);
      var newmover = findcontrol(newstate,library);
      var newscore = parseInt(findreward(role,newstate,library));
      var newcomplete = findterminalp(newstate,library);
      return makenode(newstate,newmover,newscore,newcomplete)};
  for (var i=0; i<node.actions.length; i++)
      {if (equalp(move,node.actions[i])) {return node.children[i]}}
  return node}

//==============================================================================

function processnode (node)
 {if (node.children.length===0) {expand(node)}
     else {processnode(selectnode(node))};
  updatenode(node);
  return true}

//==============================================================================

function selectnode (node)
 {var child = node.children[0];
  var score = -1;
  for (var i=0; i<node.children.length; i++)
      {var newchild = node.children[i];
       if (newchild.complete) {continue};
       var newscore = scorenode(newchild,node);
       if (newscore>score) {child = newchild; score = newscore}};
  return child}

function scorenode (node,parent)
 {var exploitation = node.utility;
  var exploration = Math.round((1 - node.visits/parent.visits)*100);
  if (parent.mover===role) {score = exploration + exploitation}
     else {score = exploration - exploitation};
  return score}

function scorenode (node,parent)
 {var exploitation = node.utility;
  var exploration = Math.round((1 - node.visits/parent.visits)*100);
  if (parent.mover===role) {score = exploration + exploitation}
     else {score = exploration};
  return score}

//==============================================================================

function expand (node)
 {node.actions = shuffle(findlegals(node.state,library));
  for (var i=0; i<node.actions.length; i++)
      {var newstate = simulate(node.actions[i],node.state,library);
       var newmover = findcontrol(newstate,library);
       var newscore = parseInt(findreward(role,newstate,library));
       var newcomplete = findterminalp(newstate,library);
       node.children[i] = makenode(newstate,newmover,newscore,newcomplete);
       if (newcomplete) {terminals++};
       nodes++};
  return true}

function shuffle (array)
 {for (var i = array.length-1; i>0; i--)
      {var j = Math.floor(Math.random() * (i + 1));
       var temp = array[i];
       array[i] = array[j];
       array[j] = temp};
  return array}

//==============================================================================

function updatenode (node)
 {node.utility = (node.mover===role) ? scoremax(node) : scoremin(node);
  node.complete = (node.mover===role) ? checkmax(node) : checkmin(node);
  node.visits = node.visits+1;
  return true}

function scoremax (node)
 {var score = node.children[0].utility;
  for (var i=1; i<node.children.length; i++)
      {var newscore = node.children[i].utility;
       if (newscore>score) {score = newscore}};
  return score}

function scoremin (node)
 {var score = node.children[0].utility;
  for (var i=1; i<node.children.length; i++)
      {var newscore = node.children[i].utility;
       if (newscore<score) {score = newscore}};
  return score}

function checkmax (node)
 {var flag = true;
  for (var i=0; i<node.children.length; i++)
      {if (!node.children[i].complete) {flag = false; continue};
       if (node.children[i].utility===100) {return true}};
  return flag}

function checkmin (node)
 {var flag = true;
  for (var i=0; i<node.children.length; i++)
      {if (!node.children[i].complete) {flag = false; continue};
       if (node.children[i].utility===0) {return true}};
  return flag}

function checkcomplete (node)
 {for (var i=0; i<node.children.length; i++)
      {if (!node.children[i].complete) {return false}};
  return true}

//==============================================================================

function selectaction (node)
 {var action = node.actions[0];
  var score = -1;
  for (var i=0; i<node.children.length; i++)
      {var child = node.children[i];
       if (child.complete && child.utility===100) {return node.actions[i]};
       if (child.complete && child.utility===0) {continue};
       var newscore = child.utility;
       if (newscore>score) {action = node.actions[i]; score = newscore}};
  return action}

function selectindex (node)
 {var index = 0;
  var score = -1;
  for (var i=0; i<node.children.length; i++)
      {var child = node.children[i];
       if (child.complete && child.utility===100) {return node.actions[i]};
       if (child.complete && child.utility===0) {continue};
       var newscore = child.utility;
       if (newscore>score) {index = i; score = newscore}};
  return index}

//==============================================================================
// test stuff - recursive selection - same as processnode but does not update
//==============================================================================

function findnode (node)
 {if (node.children.length===0) {return node};
  return findnode(selectnode(node))}

function process (node,n)
 {for (var i=0; i<n; i++) {processnode(node)};
  return true}

//==============================================================================
// End of player code
//==============================================================================
