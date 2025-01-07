from . import LanguagePatterns

class RoPatterns(LanguagePatterns):
    def init_patterns(self):
        self.patterns.update({
            'titles': [
                # Official Documents
                "PROCES VERBAL", "PROTOCOL", "ADEVERINȚĂ", "CERTIFICAT",
                "DOMNULE", "CĂTRE", "REFERAT", "RAPORT", "DECIZIE",
                "HOTĂRÂRE", "ORDIN", "DISPOZIȚIE", "CIRCULARĂ",
                "MEMORIU", "PETIȚIE", "CERERE", "CHITANȚĂ",
                
                # Historical Documents
                "HRISOV", "TESTAMENT", "ZAPIS", "IZVOD",
                "ANAFORÀ", "PITAC", "CARTE DOMNEASCĂ",
                
                # Common Headers
                "CONFIDENȚIAL", "URGENT", "STRICT SECRET",
                "ÎN ATENȚIA", "SPRE ȘTIINȚĂ", "COPIE",
                "ORIGINAL", "DUPLICAT"
            ],
            
            'honorifics': [
                # Modern Forms
                "D-lui", "D-na", "D-ra", "Dl.", "Dna.", "Dlui.", "Dnei.",
                "Domnul", "Doamna", "Domnișoara",
                "Domnului", "Doamnei", "Domnișoarei",
                
                # Historical Forms
                "Dumnealui", "Dumneaei", "Dumnealor",
                "Jupân", "Jupâneasa", "Coconul", "Cocoana",
                "Boierul", "Boieroaica", "Conu'", "Coana",
                
                # Religious/Noble Titles
                "Prea Sfințitul", "Înalt Prea Sfințitul",
                "Prea Cuviosul", "Prea Cucernicul",
                "Măria Sa", "Înălțimea Sa", "Excelența Sa",
                "Prea Înaltul", "Luminăția Sa"
            ],
            
            'positions': [
                # Administrative
                "Profesor", "Prof.", "Director", "Dir.",
                "Inspector", "Insp.", "Secretar", "Secr.",
                "Președinte", "Preș.", "Ministru", "Min.",
                "Prefect", "Primar", "Consilier", "Cons.",
                
                # Religious
                "Preot", "Pr.", "Episcop", "Ep.",
                "Mitropolit", "Arhiepiscop", "Arhim.",
                "Diacon", "Stareț", "Egumen",
                
                # Historical
                "Logofăt", "Vornic", "Vistiernic",
                "Spătar", "Postelnic", "Clucer",
                "Paharnic", "Stolnic", "Comis",
                "Pârcălab", "Ispravnic", "Zapciu",
                
                # Educational/Professional
                "Învățător", "Înv.", "Diriginte", "Dir.",
                "Rector", "Decan", "Conferențiar", "Conf.",
                "Lector", "Asistent", "Cercetător"
            ],
            
            'dates': [
                # Months - Modern
                "Ianuarie", "Ian.", "Februarie", "Febr.",
                "Martie", "Mart.", "Aprilie", "Apr.",
                "Mai", "Iunie", "Iun.", "Iulie", "Iul.",
                "August", "Aug.", "Septembrie", "Sept.",
                "Octombrie", "Oct.", "Noiembrie", "Noem.",
                "Decembrie", "Dec.",
                
                # Months - Historical
                "Gerar", "Făurar", "Mărțișor",
                "Prier", "Florar", "Cireșar",
                "Cuptor", "Gustar", "Răpciune",
                "Brumărel", "Brumar", "Undrea",
                
                # Date Formats
                r"\d{1,2}/\d{1,2}/\d{4}",  # 25/12/1900
                r"\d{1,2}\.\d{1,2}\.\d{4}",  # 25.12.1900
                
                # Historical Date References
                "Leat", "În anul", "La anul",
                "De la Hristos", "În zilele", "La văleat"
            ],
            
            'numbers': [
                # Basic Numbering
                "Nr.", "No.", "Numărul", "Pag.", "Pagina",
                "Foaia", "Fila", "Poziția", "Poz.",
                
                # Roman Numerals
                "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
                "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
                
                # Historical Number Formats
                r"\d{1,2}/\d{4}",  # 5/1892
                r"\d{1,2}\.\d{1,2}\.\d{4}",  # 15.03.1892
                
                # Reference Numbers
                "Dosar nr.", "Registru nr.", "Act nr.",
                "Poziția nr.", "Inventar nr."
            ],
            
            'locations': [
                # Modern Administrative
                "București", "Bucuresci", "Comuna", "Com.",
                "Județul", "Jud.", "Plasa", "Pl.",
                "Strada", "Str.", "District", "Dist.",
                "Sector", "Sect.", "Cartier", "Cart.",
                
                # Historical Administrative
                "Ținutul", "Țin.", "Ocol", "Oc.",
                "Plasă", "Pl.", "Raion", "Raionul",
                "Târg", "Târgul", "Sat", "Satul",
                
                # Building/Location Types
                "Mănăstirea", "Măn.", "Biserica", "Bis.",
                "Școala", "Șc.", "Spitalul", "Spit.",
                "Primăria", "Prim.", "Prefectura", "Pref."
            ],
            
            'signatures': [
                # Signature Lines
                "Semnătura", "Semnat", "ss.", "m.p.",
                "(ss)", "(m.p.)", "Pentru conformitate",
                
                # Authentication
                "Văzut", "Verificat", "Certificat",
                "Legalizat", "Autentificat",
                
                # Historical
                "Adeverim", "Întărim", "Mărturisim",
                "Cu credință", "Spre credință",
                "Pecetluit", "Cu pecete domnească"
            ],
            
            'currency': [
                # Modern Currency
                "Lei", "Bani", "L.", "b.", 
                r"\d+ Lei", r"\d+ Bani",
                "RON", "ROL",
                
                # Historical Currency
                "Galbeni", "Taleri", "Groși",
                "Parale", "Aspri", "Ducați",
                "Złoți", "Crăițari", "Sfanți"
            ],
            
            'common_abbrev': [
                # General Abbreviations
                "etc.", "ș.a.", "ș.a.m.d.", "resp.",
                "urm.", "conf.", "cf.", "op. cit.",
                "ibid.", "loc. cit.", "viz.", "v.",
                
                # Administrative
                "adr.", "adm.", "dir.", "sec.",
                "reg.", "ref.", "doc.", "dos.",
                
                # Historical/Religious
                "sf.", "cuv.", "pr.", "ep.",
                "mit.", "mân.", "bis.", "paroh.",
                
                # Measurements
                "km.", "m.", "cm.", "mm.",
                "kg.", "g.", "l.", "ha."
            ]
        })
