from pathlib import Path
from zipfile import ZipFile

from huggingface_hub import snapshot_download

SHAPENET_REPOSITORY = "ShapeNet/ShapeNetCore"


def download_synset(
    synset: str, data_directory: Path, limit: int | None = None
) -> None:
    archive_directory = data_directory / "archives"
    raw_directory = data_directory / "raw" / synset

    snapshot_download(
        repo_id=SHAPENET_REPOSITORY,
        repo_type="dataset",
        allow_patterns=[f"{synset}.zip"],
        local_dir=archive_directory,
    )

    archive = archive_directory / f"{synset}.zip"
    with ZipFile(archive) as zip_file:
        members = sorted(
            member
            for member in zip_file.namelist()
            if member.endswith("/models/model_normalized.obj")
        )
        if limit is not None:
            members = members[:limit]
        zip_file.extractall(raw_directory, members)
