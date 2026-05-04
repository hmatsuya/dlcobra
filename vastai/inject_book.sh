#!/bin/bash
# Injects a book file into an existing Docker image under a named destination.
#
# Usage (from repo root):
#   bash vastai/inject_book.sh <book.bin> <dest_name> [image:tag]
#
# Arguments:
#   book.bin    Local path to the book file
#   dest_name   Filename inside /workspace/model/ (e.g. mafu_book.bin or cobra_book.bin)
#   image:tag   Docker image to inject into (default: hmatsuya/dlshogi-usi:latest)
#
# Examples:
#   bash vastai/inject_book.sh /tmp/mafu_book.bin mafu_book.bin hmatsuya/dlshogi-usi:latest
#   bash vastai/inject_book.sh /path/to/user_book1.bin cobra_book.bin hmatsuya/dlshogi-usi:latest

set -e

BOOK_PATH="${1}"
DEST_NAME="${2}"
IMAGE="${3:-hmatsuya/dlshogi-usi:latest}"

if [ -z "$BOOK_PATH" ] || [ -z "$DEST_NAME" ]; then
    echo "Usage: bash vastai/inject_book.sh <book.bin> <dest_name> [image:tag]"
    exit 1
fi

if [ ! -f "$BOOK_PATH" ]; then
    echo "ERROR: book file not found: $BOOK_PATH"
    exit 1
fi

echo "=== Inject book into image: $IMAGE ==="
echo "  book : $BOOK_PATH ($(du -sh "$BOOK_PATH" | cut -f1))"
echo "  dest : /workspace/model/$DEST_NAME"

CONTAINER_ID=$(docker create "$IMAGE")
docker cp "$BOOK_PATH" "${CONTAINER_ID}:/workspace/model/${DEST_NAME}"
docker commit \
    --message "inject ${DEST_NAME} from $(basename "$BOOK_PATH")" \
    "$CONTAINER_ID" "$IMAGE"
docker rm "$CONTAINER_ID"

echo ""
echo "=== Done ==="
echo "  Image $IMAGE now contains /workspace/model/${DEST_NAME}"
echo ""
echo "Next step — push to Docker Hub:"
echo "  docker push $IMAGE"
