//==============================================================================
// tictactoe
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

//function renderstate (state)
// {var role = compfindx('R',seq('control','R'),state,library);
//var table = document.createElement('table');
//table.setAttribute('border','0');
//var row = table.insertRow(table.rows.length);
//var cell = row.insertCell(0);
//var board = renderboard(state);
//cell.appendChild(board);
//row = table.insertRow(table.rows.length);
//var cell = row.insertCell(0);
//cell.setAttribute('align','center');
//cell.setAttribute('style','font-size:20px');
//if (compfindp('terminal',state,library))
// {cell.innerHTML = 'Game over'}
//else {cell.innerHTML = 'Control:  ' + role};
//return table}

function renderstate(state) {
    // Encontrar el rol que tiene el control del juego
    var role = compfindx('R', seq('control', 'R'), state, library);

    // Mostrar el tablero del juego de manera simple
    console.log("Estado actual del tablero:");
    console.log(renderboard(state)); // Asumiendo que renderboard genera una representación del tablero

    // Mostrar el estado del control
    if (compfindp('terminal', state, library)) {
        console.log('El juego ha terminado.');
    } else {
        console.log('Turno de control: ' + role);
    }
}

function renderboard(state) {
    var table = document.createElement('table');
    table.setAttribute('cellspacing', '0');
    table.setAttribute('bgcolor', 'white');
    table.setAttribute('border', '10');
    makerow(table, 0, state);
    makerow(table, 1, state);
    makerow(table, 2, state);
    return table
}

function makerow(table, rownum, state) {
    var row = table.insertRow(rownum);
    makecell(row, rownum, 0, state);
    makecell(row, rownum, 1, state);
    makecell(row, rownum, 2, state);
    return row
}

function makecell(row, rownum, colnum, state) {
    var cell = row.insertCell(colnum);
    cell.setAttribute('width', '60');
    cell.setAttribute('height', '60');
    cell.setAttribute('align', 'center');
    cell.setAttribute('valign', 'center');
    cell.setAttribute('style', 'font-family:helvetica;font-size:28pt');
    rownum = (rownum + 1).toString();
    colnum = (colnum + 1).toString();
    var mark = compfindx('Z', seq('cell', rownum, colnum, 'Z'), state, seq());
    if (mark && mark != 'b') { cell.innerHTML = mark } else { cell.innerHTML = '&nbsp;' };
    return cell
}

module.exports = {
    renderstate,
    renderboard,
    makerow,
    makecell,
    makerow,
};


//==============================================================================
//==============================================================================
//==============================================================================