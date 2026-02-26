# Backlog

* bug: cannot interrupt / ctrl-c when a llm response is outstanding - wait for N seconds before giving up and quitting
* improvement: create seperate read/write(/execute?) allow lists for filesystem and shell tasks. Read permissions could be system wide by write restricted to current working directory
* feature: add a ask_permission tool where the agent can explicitly confirm an action with the user
* feature: add a presentation layer on top of agent responses to the user to properly format newlines


# Completed

* improvement: Create sqlite db in ${HOME}/.local/germ/sessions.db instead of ${CWD}/storage/orchestrator.db