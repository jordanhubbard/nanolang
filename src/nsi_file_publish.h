#ifndef NL_NSI_FILE_PUBLISH_H
#define NL_NSI_FILE_PUBLISH_H
#include "nsi_file_binding.h"

typedef enum {
    NL_FILE_PUBLISH_NONE, NL_FILE_PUBLISH_VALIDATE,
    NL_FILE_PUBLISH_PARENT, NL_FILE_PUBLISH_STAGE,
    NL_FILE_PUBLISH_INTERFACE, NL_FILE_PUBLISH_SOURCE,
    NL_FILE_PUBLISH_STAGE_SYNC, NL_FILE_PUBLISH_RENAME,
    NL_FILE_PUBLISH_PARENT_SYNC, NL_FILE_PUBLISH_CLOSE
} NlFilePublishStage;

typedef struct {
    NlFileBindingStatus status;
    NlFilePublishStage failed_stage;
    int first_errno;
    int cleanup_errno;
    bool published;
    bool durable;
    bool cleanup_pending;
    char staging_name[96];
} NlFileBindingPublishReport;

/* I publish only an authentic immutable plan. Report/input storage is disjoint.
 * The caller keeps the parent and ancestors trusted/stable and does not alter
 * another publisher's staging entries. I am not a pathname sandbox.
 * I initialize a non-NULL report even on refusal. No close is retried, no
 * existing destination is replaced, and postcommit failure never rolls back.
 * published/durable must be read independently of status before any retry.
 * I allocate no project heap storage and retain no caller pointer. */
NlFileBindingStatus nl_file_binding_publish(const NlFileBindingPlan *plan,
    const char *directory, NlFileBindingPublishReport *report);
#endif
