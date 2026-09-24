import lldb
pending={}
def enter(frame, loc, _):
    target=frame.GetThread().GetProcess().GetTarget()
    addr=frame.FindRegister('x30').GetValueAsUnsigned()
    if addr not in pending:
        bp=target.BreakpointCreateByAddress(addr)
        bp.SetScriptCallbackFunction('pr522_emit_trace.leave')
        pending[addr]=bp.GetID()
    return False
def leave(frame,loc,_):
    error=lldb.SBError()
    text=frame.GetThread().GetProcess().ReadCStringFromMemory(frame.FindRegister('x0').GetValueAsUnsigned(),2048,error)
    print('EMITTED',repr(text))
    return False
