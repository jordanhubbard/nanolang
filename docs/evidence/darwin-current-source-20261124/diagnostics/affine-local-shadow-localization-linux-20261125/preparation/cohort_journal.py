"""I retain ordered command updates and publish the original terminal JSON once."""
from pathlib import Path
import hashlib
import json
import os
import tempfile


def encoded(value, *, pretty=False):
    return (json.dumps(value, indent=2 if pretty else None)+'\n').encode()


def freeze(value):
    if isinstance(value, dict):return FrozenDict({k:freeze(v) for k,v in value.items()})
    if isinstance(value, (list, tuple)):return tuple(freeze(v) for v in value)
    return value


class FrozenDict(dict):
    def refused(self, *args, **kwargs):raise TypeError('I preserve sealed historical command records')
    __setitem__=__delitem__=clear=pop=popitem=setdefault=update=__ior__=refused


class Command(dict):
    sealed=False
    def seal(self):
        if self.sealed:return
        for key,value in self.items():dict.__setitem__(self,key,freeze(value))
        self.sealed=True
    def check(self):
        if self.sealed:raise TypeError('I preserve sealed historical command records')
    def __setitem__(self,key,value):self.check();dict.__setitem__(self,key,value)
    def __delitem__(self,key):self.check();dict.__delitem__(self,key)
    def update(self,*args,**kwargs):self.check();dict.update(self,*args,**kwargs)
    def clear(self):self.check();dict.clear(self)
    def pop(self,*args):self.check();return dict.pop(self,*args)
    def popitem(self):self.check();return dict.popitem(self)
    def setdefault(self,*args):self.check();return dict.setdefault(self,*args)
    def __ior__(self,other):self.update(other);return self


class History(list):
    def append(self,row):
        if not isinstance(row,Command):raise TypeError('I require a tracked command record')
        if self:self[-1].seal()
        list.append(self,row)
    def refused(self,*args,**kwargs):raise TypeError('I append command history without replacing prior rows')
    __setitem__=__delitem__=extend=insert=pop=clear=remove=__iadd__=__imul__=reverse=sort=refused


def atomic(path,raw):
    path=Path(path)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.pending',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(raw)
        os.replace(name,path)
    finally:
        if Path(name).exists():Path(name).unlink()


class CohortJournal:
    def __init__(self,directory,state):
        self.directory=Path(directory);self.state=state
        assert state['terminal'] is False and state['commands']==[]
        self.history=History();state['commands']=self.history
        self.path=self.directory/'cohort-journal.jsonl';self.path.open('xb').close()
        self.sequence=0;self.last_meta=None;self.last_tail=None;self.committed=False;self.last_number=0
        atomic(self.directory/'cohort-result.json',encoded(state,pretty=True))
    @staticmethod
    def row(**values):return Command(values)
    def save(self, *, metadata=False):
        assert not self.committed and self.state['commands'] is self.history
        assert len(self.history) in (self.last_number,self.last_number+1), 'I require a save for every appended command'
        events=[]
        if self.history:
            row=self.history[-1];assert row['number']==len(self.history)
            tail=encoded(row)
            if tail!=self.last_tail:
                events.append(dict(kind='command',number=len(self.history),value=row));self.last_tail=tail
        if metadata or self.last_meta is None or self.state['terminal']:
            values=dict(self.state);values['commands']=[];raw_meta=encoded(values)
            if raw_meta!=self.last_meta:
                events.append(dict(kind='metadata',value=values));self.last_meta=raw_meta
        terminal=self.state['terminal']
        if terminal:
            if self.history:self.history[-1].seal()
            final=encoded(self.state,pretty=True)
            events.append(dict(kind='commit',commands=len(self.history),sha256=hashlib.sha256(final).hexdigest(),bytes=len(final)))
        with self.path.open('ab') as stream:
            for event in events:
                self.sequence+=1;stream.write(encoded(dict(sequence=self.sequence,**event)))
        self.last_number=len(self.history)
        if terminal:
            atomic(self.directory/'cohort-result.json',final);self.committed=True
        atomic(self.directory/'cohort-progress.json',encoded(dict(terminal=terminal,commands=len(self.history),events=self.sequence,complete_terminal_file=terminal)))


def replay(path, *, allow_incomplete=False):
    raw=Path(path).read_bytes();parts=raw.splitlines(keepends=True)
    if parts and not parts[-1].endswith(b'\n'):
        if not allow_incomplete:raise ValueError('I require a complete journal line')
        parts.pop()
    state=None;rows=[];committed=False;commit=None
    for number,line in enumerate(parts,1):
        event=json.loads(line);assert event['sequence']==number and not committed
        kind=event['kind']
        if kind=='metadata':
            assert event['value']['commands']==[];state=event['value']
        elif kind=='command':
            index=event['number'];assert index==event['value']['number']
            assert index in (len(rows),len(rows)+1) and index>0
            if index==len(rows)+1:rows.append(event['value'])
            else:rows[-1]=event['value']
        elif kind=='commit':
            assert state is not None and state['terminal'] and event['commands']==len(rows)
            state['commands']=rows;final=encoded(state,pretty=True)
            assert len(final)==event['bytes'] and hashlib.sha256(final).hexdigest()==event['sha256']
            committed=True;commit=event
        else:raise ValueError('I require a known journal event')
    if state is None:raise ValueError('I require retained cohort metadata')
    state['commands']=rows
    if not committed:state['terminal']=False
    return state,dict(committed=committed,events=len(parts),commit=commit)
