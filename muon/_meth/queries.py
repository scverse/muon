# Fetches genomic info from public API endpoints

from operator import index
import natsort
import requests
import pandas as pd

ENSEMBL_BASE_URL = "https://rest.ensembl.org"
UCSC_BASE_URL = "https://api.genome.ucsc.edu"


def fetch_endpoint(server, request, content_type):
    """
    Fetch an endpoint from the server, allow overriding of default content-type
    """
    r = requests.get(f"{server}/{request}", headers={"Accept": content_type})

    if not r.ok:
        r.raise_for_status()

    if content_type == "application/json":
        return r.json()
    else:
        return r.text


class Ensembl:
    """Provides access to the ENSEMBL public api"""

    def __init__(self, server=ENSEMBL_BASE_URL):
        self.server = server

    def fetch_endpoint(self, request, content_type):
        return fetch_endpoint(server=self.server, request=request, content_type=content_type)

    def list_species(self, all_columns=False):
        """List available genomes from ENSEMBL."""
        request = "info/species"
        o = self.fetch_endpoint(request=request, content_type="application/json")
        df = pd.DataFrame(o["species"])
        if all_columns:
            df = pd.DataFrame.from_records(o["species"], index="name")
        else:
            df = pd.DataFrame.from_records(
                o["species"], index="name", columns=["name", "display_name"]
            )

        return df

    def get_chromosome_sizes(self, species, chromosomes_only=True):
        """Fetch chromosome sizes for the specified genome."""
        request = f"info/assembly/{species}"
        o = self.fetch_endpoint(request=request, content_type="application/json")
        df = pd.DataFrame.from_records(
            o["top_level_region"], columns=("name", "coord_system", "length"), index="name"
        )
        if chromosomes_only:
            df = df.loc[df.coord_system == "chromosome"]
        return df.sort_values(by="name", key=natsort.natsort_keygen(), ignore_index=True)


class Ucsc:
    """Provides access to the UCSC public api"""

    def __init__(self, server=UCSC_BASE_URL):
        self.server = server

    def fetch_endpoint(self, request, content_type):
        return fetch_endpoint(server=self.server, request=request, content_type=content_type)

    def list_genomes(self, all_columns=False):
        """List available genomes from UCSC."""
        request = "list/ucscGenomes"
        o = self.fetch_endpoint(request=request, content_type="application/json")
        if all_columns:
            df = pd.DataFrame.from_dict(o["ucscGenomes"], orient="index")
        else:
            df = pd.DataFrame.from_dict(
                o["ucscGenomes"], orient="index", columns=("organism", "scientificName")
            )
        return df

    def get_chromosome_sizes(self, genome, chromosomes_only=True):
        """Fetch chromosome sizes for the specified genome."""
        request = f"list/chromosomes?genome={genome}"
        o = self.fetch_endpoint(request=request, content_type="application/json")
        df = pd.DataFrame.from_dict(o["chromosomes"], columns=["length"], orient="index")

        # Trying to determine which chromosomes are "standard chromosomes"
        df["coord_system"] = "chromosome"
        df.loc[df.index.str.contains("_"), "coord_system"] = "scaffold"
        if chromosomes_only:
            df = df.loc[df.coord_system == "chromosome"]
        return df.sort_index(key=natsort.natsort_keygen())


#
