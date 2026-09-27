import lldb, struct
pending = {}
def enter(frame, loc, _):
    process = frame.GetThread().GetProcess()
    error = lldb.SBError()
    data = process.ReadMemory(frame.FindRegister('x1').GetValueAsUnsigned(), 32, error)
    call = struct.unpack('<8I', data) if error.Success() else ()
    bp = process.GetTarget().BreakpointCreateByAddress(frame.FindRegister('x30').GetValueAsUnsigned())
    bp.SetOneShot(True)
    pending[bp.GetID()] = call
    bp.SetScriptCallbackFunction('pr522_owner_trace.leave')
    return False
def leave(frame, loc, _):
    call = pending.pop(loc.GetBreakpoint().GetID(), ())
    print('CALL', call, 'INDEX', frame.FindRegister('x0').GetValueAsSigned())
    return False
