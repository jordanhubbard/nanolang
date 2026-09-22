/* I share the reviewed fresh-image worker and bounded diagnostic contract. */
#ifndef NANO_STRUCT_OWNERSHIP_WORKER_H
#define NANO_STRUCT_OWNERSHIP_WORKER_H
extern char **environ;
static const char *fixture_executable;
static pid_t spawn_fault_case(int errors[2],int lookup,int kind,size_t position,int mode) {
    char kind_text[16],position_text[32],mode_text[16];
    snprintf(kind_text,sizeof(kind_text),"%d",kind);
    snprintf(position_text,sizeof(position_text),"%zu",position);
    snprintf(mode_text,sizeof(mode_text),"%d",mode);
    char *lookup_args[]={(char *)fixture_executable,"_lookup_fault",kind_text,position_text,mode_text,NULL};
    char *snapshot_args[]={(char *)fixture_executable,"_snapshot_fault",position_text,mode_text,NULL};
    posix_spawn_file_actions_t actions;
    assert(posix_spawn_file_actions_init(&actions)==0);
    assert(posix_spawn_file_actions_adddup2(&actions,errors[1],STDERR_FILENO)==0);
    assert(posix_spawn_file_actions_addclose(&actions,errors[0])==0);
    if (errors[1]!=STDERR_FILENO) assert(posix_spawn_file_actions_addclose(&actions,errors[1])==0);
    pid_t child;
    int result=posix_spawn(&child,fixture_executable,&actions,NULL,
        lookup ? lookup_args : snapshot_args,environ);
    assert(posix_spawn_file_actions_destroy(&actions)==0);
    if (result) fprintf(stderr,"I could not spawn my ownership fault worker: %s\n",strerror(result));
    assert(result==0); return child;
}
static int fixture_number(const char *text,unsigned maximum,unsigned *value) {
    if (!text || !*text) return 0;
    unsigned result=0;
    for (const char *p=text;*p;++p) {
        if (*p<'0' || *p>'9') return 0;
        unsigned digit=(unsigned)(*p-'0');
        if (digit>maximum || result>(maximum-digit)/10) return 0;
        result=result*10+digit;
    }
    *value=result; return 1;
}

static void read_child_diagnostic(int fd,pid_t child,const char *expected) {
    enum { LIMIT=65536 }; char message[LIMIT+1]; size_t used=0;
    for (;;) {
        char chunk[1024]; ssize_t got=read(fd,chunk,sizeof(chunk));
        if (got<0 && errno==EINTR) continue;
        assert(got>=0); if (!got) break;
        if ((size_t)got>LIMIT-used) {
            fprintf(stderr,"I retained the first64KiB of an oversized child diagnostic:\n");
            fwrite(message,1,used,stderr); fwrite(chunk,1,LIMIT-used,stderr); fflush(stderr);
            close(fd); (void)kill(child,SIGKILL); int status;
            while (waitpid(child,&status,0)<0 && errno==EINTR) {}
            assert(!"child diagnostic exceeded the retained bound");
        }
        memcpy(message+used,chunk,(size_t)got); used+=(size_t)got;
    }
    message[used]='\0'; close(fd);
    if (strcmp(message,expected)) {
        fprintf(stderr,"I retained an unexpected child diagnostic (%zu bytes):\n",used);
        fwrite(message,1,used,stderr); fflush(stderr);
        int status; while (waitpid(child,&status,0)<0 && errno==EINTR) {}
    }
    assert(!strcmp(message,expected));
}

#endif
