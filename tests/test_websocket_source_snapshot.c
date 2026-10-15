/* I retain exact WebSocket companions independently of later filesystem changes. */
#include "nanoisa/file_source_snapshot.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc,char **argv) {
    assert(argc==2);const char *origin=argv[1];
    NlFileSourceSnapshots *snapshots=NULL;
    assert(nl_file_source_snapshots_new(&snapshots)==NL_BINDING_OK);
    size_t index=99;
    assert(nl_service_source_snapshot_open(snapshots,3,origin,strlen(origin),"websocket.json",14,&index)==NL_BINDING_OK);
    assert(index==0 && nl_service_source_snapshot_catalog(snapshots,index)==3);
    size_t size=0;
    const unsigned char *source=nl_file_source_snapshot_bytes(snapshots,index,3,&size);
    const char expected[]="# I declare catalog1; this declaration alone grants no host authority.\nservice \"nsi:nanolang/websocket\" catalog 1 from \"interface.nsi.json\"\n";
    assert(source && size==sizeof expected-1 && !memcmp(source,expected,size));
    size_t canonical_size=0;
    const unsigned char *canonical=nl_file_source_snapshot_bytes(snapshots,index,2,&canonical_size);
    assert(canonical && canonical_size>0);
    unsigned char *saved=malloc(canonical_size);assert(saved);memcpy(saved,canonical,canonical_size);
    size_t storage=nl_file_source_snapshot_storage(snapshots);
    assert(storage<=nl_file_source_snapshot_peak_bound(snapshots));
    assert(nl_file_source_snapshot_peak_bound(snapshots)<=NL_FILE_SOURCE_SNAPSHOT_BUDGET);
    index=99;
    assert(nl_service_source_snapshot_open(snapshots,2,origin,strlen(origin),"websocket.json",14,&index)==NL_BINDING_INVALID);
    assert(nl_service_source_snapshot_open(snapshots,3,origin,strlen(origin),"file.json",9,&index)==NL_BINDING_INVALID);
    assert(nl_service_source_snapshot_open(snapshots,3,origin,strlen(origin),"socket.json",11,&index)==NL_BINDING_INVALID);
    assert(nl_service_source_snapshot_open(snapshots,4,origin,strlen(origin),"missing.json",12,&index)==NL_BINDING_INVALID);
    assert(nl_service_source_snapshot_open(snapshots,3,origin,strlen(origin),"missing.json",12,&index)==NL_BINDING_IO);
    assert(index==99 && nl_file_source_snapshot_count(snapshots)==1);
    assert(nl_file_source_snapshot_storage(snapshots)==storage);
    assert(nl_service_source_snapshot_open(snapshots,1,origin,strlen(origin),"file.json",9,&index)==NL_BINDING_OK);
    assert(index==1 && nl_service_source_snapshot_catalog(snapshots,index)==1);
    assert(nl_service_source_snapshot_open(snapshots,2,origin,strlen(origin),"socket.json",11,&index)==NL_BINDING_OK);
    assert(index==2 && nl_service_source_snapshot_catalog(snapshots,index)==2);
    size_t path_size=0;const unsigned char *path=nl_file_source_snapshot_bytes(snapshots,0,0,&path_size);
    assert(path && path_size==strlen((const char *)path));assert(!unlink((const char *)path));
    assert(!memcmp(canonical,saved,canonical_size));
    assert(!memcmp(source,expected,sizeof expected-1));
    index=99;
    assert(nl_service_source_snapshot_open(snapshots,3,origin,strlen(origin),"websocket.json",14,&index)==NL_BINDING_IO);
    assert(index==99 && nl_file_source_snapshot_count(snapshots)==3);
    free(saved);nl_file_source_snapshots_free(snapshots);puts("PASS immutable WebSocket source snapshots");return 0;
}
