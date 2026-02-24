# Backlog

* bug: cannot interrupt / ctrl-c when a llm response is outstanding - wait for N seconds before giving up and quitting
* improvement: create seperate read/write(/execute?) allow lists for filesystem and shell tasks. Read permissions could be wider than write.
* improvement: gracefully exit when ctrl-c pressed in stead of showing a stack trace

# Completed

* improvement: Create sqlite db in ${HOME}/.local/germ/sessions.db instead of ${CWD}/storage/orchestrator.db