#!/usr/bin/env bash

set -euo pipefail

target_folder="${1:-}"
quota="${2:-}"

help() {
    cat << _EOF
Delete files from folder <target_folder>
until disk usage is below the <quota> percentage.

${0} <target_folder> <quota>

ARGS:
    <target_folder>
            Folder to check
    <quota>
            Delete files until disk usage is below this percentage
_EOF
}

if [[ -z "${target_folder}" ]]; then
    echo "Please provide a target folder"
    help
    exit 1
elif [[ ! -d "${target_folder}" ]]; then
    echo "Expecting a directory"
    help
    exit 1
fi

if [[ -z "${quota}" ]]; then
    echo "Please provide a quota for target folder"
    help
    exit 1
elif [[ ! "${quota}" =~ ^[0-9]+$ ]]; then
    echo "Expecting a number between 0-99"
    help
    exit 1
elif (( quota > 99 )); then
    echo "Expecting a number between 0-99"
    help
    exit 1
fi


current_disk_usage="$(df "${target_folder}" | awk 'NR==2 {gsub("%",""); print $5}')"

if (( quota > current_disk_usage )); then
    echo "Nothing to do. Quota (${quota}) not reached (${current_disk_usage})"
    exit 0
else
    while (( quota < current_disk_usage )); do
        sorted_files="$(find "${target_folder}" -type f -printf '%T@ %p\n' | sort -nk1)"
        oldest_file="$(printf "%s\n" "${sorted_files}" | awk 'NR==1 {$1=""; sub(/^ /,""); print}')"
        if [[ -z "${oldest_file}" ]]; then
            echo "No files can be deleted as the directory is empty: ${target_folder}"
            echo "Disk usage coming from somewhere else."
            exit 1
        fi

        if ! rm --verbose "${oldest_file}"; then
            echo "Cannot delete file: ${oldest_file}"
            exit 1
        fi

        current_disk_usage="$(df "${target_folder}" | awk 'NR==2 {gsub("%",""); print $5}')"
    done
fi
