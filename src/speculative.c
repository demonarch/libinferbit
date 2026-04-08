#include "inferbit_internal.h"

/* TODO: Milestone later — speculative decoding */

void inferbit_set_draft_model(inferbit_model* model, inferbit_model* draft, int draft_tokens) {
    if (!model) return;
    model->draft_model  = draft;
    model->draft_tokens = draft_tokens > 0 ? draft_tokens : 4;
}

void inferbit_unset_draft_model(inferbit_model* model) {
    if (!model) return;
    model->draft_model  = NULL;
    model->draft_tokens = 0;
}
