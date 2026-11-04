"""Joins: no row multiplication, no silent loss, unmatched keys reported."""
import pandas as pd
import pytest

from churn_pipeline.events import ingest_pending
from churn_pipeline.merge import build_customer_master
from churn_pipeline.sources import load_customers, load_profiles
from churn_pipeline.storage import connect, replace_table


@pytest.fixture
def loaded(paths):
    customers = load_customers(paths.ibm_csv).valid
    profiles = load_profiles(paths.profiles_json).valid
    conn = connect(paths.sqlite_db)
    replace_table(conn, "customers", customers, pk="customer_id")
    ingest_pending(conn, paths.events_dir, "r")
    yield customers, profiles, conn
    conn.close()


def test_master_has_one_row_per_customer_and_reports_unmatched(loaded):
    customers, profiles, conn = loaded
    master, rep = build_customer_master(customers, profiles, conn)

    assert len(master) == 3 and master["customer_id"].is_unique
    assert rep.customers_with_profile == 2
    assert rep.customers_without_profile == 1          # 2222 kept, fields null
    assert rep.orphan_profile_ids == ["9999-ZZZZZ"]    # reported, not in master
    assert rep.orphan_events == 1 and rep.orphan_event_customers == 1
    assert "9999-ZZZZZ" not in set(master["customer_id"])

    row = master.set_index("customer_id").loc["1111-AAAAA"]
    assert (row["n_events"], row["n_login"], row["n_support_call"]) == (2, 1, 1)
    assert row["total_duration_minutes"] == 12.5

    no_events = master.set_index("customer_id").loc["2222-BBBBB"]
    assert no_events["n_events"] == 0 and pd.isna(no_events["last_event_ts"])
    assert not no_events["has_profile"]


def test_duplicate_profiles_would_multiply_rows_and_are_blocked(loaded):
    customers, profiles, conn = loaded
    dup = pd.concat([profiles, profiles.iloc[[0]]], ignore_index=True)

    # The trap: an unvalidated left join silently grows the table.
    naive = customers.merge(dup, on="customer_id", how="left")
    assert len(naive) == len(customers) + 1

    # The guard: validate="one_to_one" turns that into a loud failure.
    with pytest.raises(pd.errors.MergeError):
        build_customer_master(customers, dup, conn)


def test_inner_join_would_silently_lose_labelled_customers(loaded):
    customers, profiles, _ = loaded
    inner = customers.merge(profiles, on="customer_id", how="inner")
    assert len(inner) < len(customers)  # why the master uses a left join
