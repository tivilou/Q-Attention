#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

readonly REPO_ROOT=/home/Q-Attention/qattn-artifact-exchange
readonly SERVICE_NAME=qattn-artifact-exchange-18084
readonly SERVICE_USER=qattnexchange
readonly APP_ROOT=/opt/qattn-artifact-exchange
readonly STORAGE_ROOT=/srv/qattn-artifact-exchange/storage
readonly SECRET_ROOT=/etc/qattn-artifact-exchange
readonly LOG_ROOT=/var/log/qattn-artifact-exchange
readonly SUPERVISOR_FILE=/etc/supervisor/conf.d/${SERVICE_NAME}.conf
readonly PUBLIC_HOST=117.50.198.37
readonly PORT=18084

[[ ${EUID} -eq 0 ]] || { echo 'run as root' >&2; exit 1; }
[[ -f "$REPO_ROOT/scripts/collab/server/qattn_exchange_server.py" ]] || { echo 'service source missing' >&2; exit 1; }

getent group "$SERVICE_USER" >/dev/null || groupadd --system "$SERVICE_USER"
getent passwd "$SERVICE_USER" >/dev/null || useradd --system --gid "$SERVICE_USER" --home-dir /nonexistent --no-create-home --shell /usr/sbin/nologin "$SERVICE_USER"

install -d -o root -g root -m 0755 "$APP_ROOT" "$APP_ROOT/web"
install -d -o "$SERVICE_USER" -g "$SERVICE_USER" -m 0750 "$STORAGE_ROOT" "$LOG_ROOT"
install -d -o root -g "$SERVICE_USER" -m 0750 "$SECRET_ROOT"
install -m 0644 -o root -g root "$REPO_ROOT/scripts/collab/server/qattn_exchange_server.py" "$APP_ROOT/qattn_exchange_server.py"
install -m 0644 -o root -g root "$REPO_ROOT/scripts/collab/server/web"/* "$APP_ROOT/web/"

readonly TOKEN_FILE=$SECRET_ROOT/token
readonly CA_KEY=$SECRET_ROOT/exchange-ca.key
readonly CA_CERT=$SECRET_ROOT/exchange-ca.crt
readonly SERVER_KEY=$SECRET_ROOT/server.key
readonly SERVER_CERT=$SECRET_ROOT/server.crt
readonly CSR_FILE=$SECRET_ROOT/server.csr
readonly EXT_FILE=$SECRET_ROOT/server.ext

if [[ ! -s "$TOKEN_FILE" ]]; then
    openssl rand -hex 48 > "$TOKEN_FILE"
fi
chown root:"$SERVICE_USER" "$TOKEN_FILE"
chmod 0440 "$TOKEN_FILE"

if [[ ! -s "$CA_KEY" || ! -s "$CA_CERT" ]]; then
    openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:3072 -out "$CA_KEY"
    openssl req -x509 -new -sha256 -days 3650 -key "$CA_KEY" -out "$CA_CERT" \
        -subj '/CN=Q-Attention Artifact Exchange CA' \
        -addext 'basicConstraints=critical,CA:TRUE' \
        -addext 'keyUsage=critical,keyCertSign,cRLSign'
fi

if [[ ! -s "$SERVER_KEY" || ! -s "$SERVER_CERT" ]]; then
    openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out "$SERVER_KEY"
    openssl req -new -sha256 -key "$SERVER_KEY" -out "$CSR_FILE" -subj "/CN=$PUBLIC_HOST"
    printf '%s\n' \
        'basicConstraints=critical,CA:FALSE' \
        'keyUsage=critical,digitalSignature,keyEncipherment' \
        'extendedKeyUsage=serverAuth' \
        "subjectAltName=IP:$PUBLIC_HOST" > "$EXT_FILE"
    openssl x509 -req -in "$CSR_FILE" -CA "$CA_CERT" -CAkey "$CA_KEY" \
        -CAcreateserial -out "$SERVER_CERT" -days 825 -sha256 -extfile "$EXT_FILE"
fi

chown root:"$SERVICE_USER" "$CA_CERT" "$SERVER_CERT" "$SERVER_KEY"
chmod 0640 "$CA_CERT" "$SERVER_CERT" "$SERVER_KEY"
chmod 0600 "$CA_KEY"
install -d -o root -g root -m 0755 "$REPO_ROOT/scripts/collab/certs"
install -m 0644 -o root -g root "$CA_CERT" "$REPO_ROOT/scripts/collab/certs/qattn-exchange-ca.crt"

cat > "$SUPERVISOR_FILE" <<EOF
[program:$SERVICE_NAME]
command=/usr/bin/python3 $APP_ROOT/qattn_exchange_server.py --bind 0.0.0.0 --port $PORT --root $STORAGE_ROOT --web-root $APP_ROOT/web --token-file $TOKEN_FILE --max-bytes 1073741824 --tls-cert $SERVER_CERT --tls-key $SERVER_KEY
directory=$APP_ROOT
user=$SERVICE_USER
autostart=true
autorestart=unexpected
startsecs=3
stopsignal=TERM
stopwaitsecs=30
redirect_stderr=true
stdout_logfile=$LOG_ROOT/supervisor.log
stdout_logfile_maxbytes=20MB
stdout_logfile_backups=5
environment=PYTHONUNBUFFERED="1"
EOF
chown root:root "$SUPERVISOR_FILE"
chmod 0644 "$SUPERVISOR_FILE"

supervisorctl reread
supervisorctl update "$SERVICE_NAME"
supervisorctl restart "$SERVICE_NAME" >/dev/null 2>&1 || supervisorctl start "$SERVICE_NAME"
mkdir -p "$STORAGE_ROOT/q-attention/qepvg-case-study-recovery/20260927T071350Z"
chown -R "$SERVICE_USER:$SERVICE_USER" /srv/qattn-artifact-exchange

printf 'service=%s\nurl=https://%s:%s\nnamespace=q-attention\nstorage=%s\ntoken=%s\nca=%s\n' \
    "$SERVICE_NAME" "$PUBLIC_HOST" "$PORT" "$STORAGE_ROOT" "$TOKEN_FILE" \
    "$REPO_ROOT/scripts/collab/certs/qattn-exchange-ca.crt"
