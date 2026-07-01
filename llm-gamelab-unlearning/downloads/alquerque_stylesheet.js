//==============================================================================
// alquerque
//==============================================================================
const {
    acons,
    adjoin,
    adjoinit,
    amongp,
    arg1,
    arg2,
    assoc,
    backup,
    baseapply,
    baseapplybuiltin,
    baseapplylist,
    baseapplymath,
    baseapplyrs,
    baseanswers,
    basefindg,
    basefindn,
    basefindp,
    basefinds,
    basefindx,
    basesome,
    basesomeand,
    basesomeatom,
    basesomebase,
    basesomedistinct,
    basesomeground,
    basesomenot,
    basesomeor,
    basesomesame,
    basesomeview,
    baseunindex,
    bitand,
    bitior,
    bitlsh,
    bitnot,
    bitxor,
    callconjunction,
    calldistinct,
    calleval,
    callevaluation,
    callmember,
    callnegation,
    callsame,
    car,
    cdr,
    cons,
    delistify,
    dropfact,
    eliminatefacts,
    eliminaterules,
    envlookupfacts,
    eval,
    factindexps,
    find,
    findp,
    first,
    flatindex,
    flatunindex,
    freevarsexp,
    fullindex,
    fullunindex,
    getbases,
    getdate,
    getfactarity,
    getrulearity,
    getviews,
    getyear,
    head,
    index,
    indexps,
    indexsymbol,
    insertfact,
    insertrule,
    kif,
    len,
    list,
    makedefinition,
    makeequality,
    makeinequality,
    makenegation,
    maketransition,
    numberize,
    plugvar,
    plugexp,
    remfact,
    remcontent,
    reverse,
    rplaca,
    rplacd,
    scan,
    seq,
    stripquotes,
    stringify,
    tail,
    unify,
    unindexsymbol,
    variance,
    symbolp,
    append,
    binaryappend,
    debugfindn,
    debugfindp,
    debugfinds,
    debugfindx,
    fastread,
    fastreaddata,
    fastreaditems,
    getdataset,
    getlength,
    getmonth,
    getsecond,
    grindspaces,
    hastype,
    kifexp,
    kifparenlist,
    listify,
    makeexistential,
    midrange,
    minimum,
    newsymbolize,
    read,
    readitems,
    scanstring,
    tracecall,
    traceexit,
    untrace,
    uniquify,
    zniquify,
    definemorerules,
    //compfinds,
    nil,
    nullp,
    lookuprules,
    indexees,
    compfindp,
    compfindx,
    compfinds,
    compfindn,
    compfindg,
    sortfinds,
    compvalue


} = require('../epilog');

function renderstate(state) {
    var step = compfindx('N', seq('step', 'N'), state, library);
    var role = compfindx('R', seq('control', 'R'), state, library);
    var table = document.createElement('table');
    table.setAttribute('border', '0');
    var row = table.insertRow(0);
    var cell = row.insertCell(0);
    cell.setAttribute('align', 'center');
    cell.setAttribute('style', 'font-size:20px');
    cell.innerHTML = 'Step:  ' + step;
    row = table.insertRow(1);
    var cell = row.insertCell(0);
    var board = renderboard(state);
    cell.appendChild(board);
    row = table.insertRow(2);
    var cell = row.insertCell(0);
    cell.setAttribute('align', 'center');
    cell.setAttribute('style', 'font-size:20px');
    if (compfindp('terminal', state, library)) { cell.innerHTML = 'Game over' } else { cell.innerHTML = 'Control:  ' + role };
    return table
}

function renderboard(state) {
    var canvas = document.createElement('canvas');
    canvas.setAttribute('width', '270px');
    canvas.setAttribute('height', '270px');

    drawline(15, 15, 255, 15, canvas);
    drawline(15, 75, 255, 75, canvas);
    drawline(15, 135, 255, 135, canvas);
    drawline(15, 195, 255, 195, canvas);
    drawline(15, 255, 255, 255, canvas);

    drawline(15, 15, 15, 255, canvas);
    drawline(75, 15, 75, 255, canvas);
    drawline(135, 15, 135, 255, canvas);
    drawline(195, 15, 195, 255, canvas);
    drawline(255, 15, 255, 255, canvas);

    drawline(15, 15, 255, 255, canvas);
    drawline(15, 255, 255, 15, canvas);
    drawline(135, 15, 15, 135, canvas);
    drawline(15, 135, 135, 255, canvas);
    drawline(135, 255, 255, 135, canvas);
    drawline(255, 135, 135, 15, canvas);

    drawnode('1', '1', 0, 0, state, canvas);
    drawnode('1', '2', 60, 0, state, canvas);
    drawnode('1', '3', 120, 0, state, canvas);
    drawnode('1', '4', 180, 0, state, canvas);
    drawnode('1', '5', 240, 0, state, canvas);

    drawnode('2', '1', 0, 60, state, canvas);
    drawnode('2', '2', 60, 60, state, canvas);
    drawnode('2', '3', 120, 60, state, canvas);
    drawnode('2', '4', 180, 60, state, canvas);
    drawnode('2', '5', 240, 60, state, canvas);

    drawnode('3', '1', 0, 120, state, canvas);
    drawnode('3', '2', 60, 120, state, canvas);
    drawnode('3', '3', 120, 120, state, canvas);
    drawnode('3', '4', 180, 120, state, canvas);
    drawnode('3', '5', 240, 120, state, canvas);

    drawnode('4', '1', 0, 180, state, canvas);
    drawnode('4', '2', 60, 180, state, canvas);
    drawnode('4', '3', 120, 180, state, canvas);
    drawnode('4', '4', 180, 180, state, canvas);
    drawnode('4', '5', 240, 180, state, canvas);

    drawnode('5', '1', 0, 240, state, canvas);
    drawnode('5', '2', 60, 240, state, canvas);
    drawnode('5', '3', 120, 240, state, canvas);
    drawnode('5', '4', 180, 240, state, canvas);
    drawnode('5', '5', 240, 240, state, canvas);

    return canvas
}

//------------------------------------------------------------------------------
// Drawing subroutines
//------------------------------------------------------------------------------

function drawred(x, y, w) {
    var ctx = w.getContext('2d');
    ctx.beginPath();
    ctx.lineWidth = 2;
    ctx.arc(x + 15, y + 15, 12, 0, 2 * Math.PI, false);
    ctx.stroke();
    ctx.fillStyle = "#ff8888";
    ctx.fill();
}

function drawblack(x, y, w) {
    var ctx = w.getContext('2d');
    ctx.beginPath();
    ctx.lineWidth = 2;
    ctx.arc(x + 15, y + 15, 12, 0, 2 * Math.PI, false);
    ctx.stroke();
    ctx.fillStyle = "#888888";
    ctx.fill();
}

function drawblank(x, y, w) {
    var ctx = w.getContext('2d');
    ctx.beginPath();
    ctx.lineWidth = 2;
    ctx.arc(x + 15, y + 15, 12, 0, 2 * Math.PI, false);
    ctx.stroke();
    ctx.fillStyle = "#ffffff";
    ctx.fill();
}

function drawline(u, v, x, y, w) {
    var ctx = w.getContext('2d');
    ctx.beginPath();
    ctx.lineWidth = 2;
    ctx.moveTo(u, v);
    ctx.lineTo(x, y);
    ctx.stroke()
}

function drawnode(m, n, x, y, state, canvas) {
    var mark = compfindx('X', seq('cell', m, n, 'X'), state, library);
    if (mark == 'red') { drawred(x, y, canvas); return true };
    if (mark == 'black') { drawblack(x, y, canvas); return true };
    drawblank(x, y, canvas);
    return true
}

function drawscore(text, x, y, w) {
    var ctx = w.getContext('2d');
    ctx.fillStyle = "#000000";
    ctx.font = "28px Times"
    ctx.fillText(text, x + 12, y + 18);
    return true
}


module.exports = {
    renderstate,
    renderboard
};
//==============================================================================
//==============================================================================
//==============================================================================