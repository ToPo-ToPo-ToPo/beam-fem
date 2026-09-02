import json
import zipfile

import pytest

from beamfem.io import (
    create_release_archive, restore_release_archive, verify_release_archive,
)


def test_release_archive_round_trip_and_retention(tmp_path):
    result = tmp_path / "result.json"
    report = tmp_path / "report.html"
    result.write_text('{"feasible": true}\n', encoding="utf-8")
    report.write_text("<html>verified</html>\n", encoding="utf-8")
    bundle = create_release_archive([result, report], tmp_path / "release.zip", retention_days=730)
    manifest = verify_release_archive(bundle)
    assert manifest["retention_days"] == 730
    restored = restore_release_archive(bundle, tmp_path / "rollback")
    assert [item.name for item in restored] == ["result.json", "report.html"]
    assert (tmp_path / "rollback" / "result.json").read_bytes() == result.read_bytes()


def test_release_archive_rejects_tampering(tmp_path):
    result = tmp_path / "result.json"
    result.write_text("{}", encoding="utf-8")
    bundle = create_release_archive([result], tmp_path / "release.zip")
    with pytest.warns(UserWarning, match="Duplicate"):
        with zipfile.ZipFile(bundle, "a") as archive:
            archive.writestr("result.json", b"tampered")
    with pytest.raises(ValueError, match="duplicate"):
        verify_release_archive(bundle)


def test_restore_never_overwrites_existing_artifact(tmp_path):
    source = tmp_path / "result.json"
    source.write_text("{}", encoding="utf-8")
    bundle = create_release_archive([source], tmp_path / "release.zip")
    target = tmp_path / "rollback"
    target.mkdir()
    (target / "result.json").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        restore_release_archive(bundle, target)
