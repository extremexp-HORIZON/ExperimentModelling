#! /bin/bash

CODE_SERVER_HOST="${CODE_SERVER_HOST:-localhost}"
CODE_SERVER_PORT="${CODE_SERVER_PORT:-8080}"
CODE_SERVER_ORIGIN="${CODE_SERVER_ORIGIN:-localhost}"
CODE_SERVER_WELCOME_TEXT="${CODE_SERVER_WELCOME_TEXT:-welcome}"
CODE_SERVER_APP_NAME="${CODE_SERVER_APP_NAME:-ExtremeXPDSL}"

PASSWORD=$PASSWORD code-server /home/user/workspace --bind-addr "$CODE_SERVER_HOST":$CODE_SERVER_PORT --app-name=$CODE_SERVER_APP_NAME --welcome-text=$CODE_SERVER_WELCOME_TEXT --disable-update-check --disable-workspace-trust --trusted-origins $CODE_SERVER_ORIGIN
