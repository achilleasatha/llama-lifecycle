import sqlite3


def load_data_in_chunks(
    filepath: str = "./oasst1.db",
    chunk_size: int | None = None,
):
    conn = sqlite3.connect(filepath)
    cursor = conn.cursor()

    query = """
            SELECT * FROM dataset
            WHERE lang='en' AND deleted=0 AND synthetic=0 AND tree_state='ready_for_export'
            """
    cursor.execute(query)

    while True:
        rows = cursor.fetchmany(chunk_size)
        if not rows:
            break

        yield rows

    conn.close()


# custom_filter = "
