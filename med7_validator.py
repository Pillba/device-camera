from typing import Iterable, Optional


class Med7Validator:
    """Optional Med7 medication-name extractor."""

    def __init__(self, model_name: str = "en_core_med7_lg"):
        self._nlp = None
        try:
            import spacy
        except ImportError:
            return

        try:
            self._nlp = spacy.load(model_name)
        except OSError:
            return

    @property
    def available(self) -> bool:
        return self._nlp is not None

    def extract_name(
        self, text: str, drug_names: Iterable[str]
    ) -> Optional[str]:
        if self._nlp is None:
            return None

        names = {name.casefold(): name for name in drug_names}
        document = self._nlp(text)
        for entity in document.ents:
            if entity.label_.upper() not in {"DRUG", "MEDICATION"}:
                continue
            candidate = entity.text.casefold().strip()
            if candidate in names:
                return names[candidate]
        return None
