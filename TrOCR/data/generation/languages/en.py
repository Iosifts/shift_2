from . import LanguagePatterns

class EnPatterns(LanguagePatterns):
    def init_patterns(self):
        self.patterns.update({
            'titles': [
                # Official Documents
                "MEMORANDUM", "CERTIFICATE", "AFFIDAVIT", "DECLARATION",
                "NOTICE", "REPORT", "MINUTES", "AGREEMENT",
                "DEED", "CONTRACT", "LAST WILL AND TESTAMENT",
                "TO WHOM IT MAY CONCERN", "POWER OF ATTORNEY",
                
                # Common Headers
                "RE:", "SUBJECT:", "IN RE:", "REGARDING:",
                "CONFIDENTIAL", "URGENT", "PRIVATE AND CONFIDENTIAL"
            ],
            
            'honorifics': [
                # Basic Titles
                "Mr.", "Mrs.", "Miss", "Ms.", "Mx.",
                "Dr.", "Prof.", "Rev.",
                
                # Extended Titles
                "Sir", "Madam", "Lady", "Lord",
                "His Excellency", "Her Excellency",
                "The Right Honorable", "The Honorable",
                
                # Military Ranks
                "Cpt.", "Captain", "Lt.", "Lieutenant",
                "Col.", "Colonel", "Gen.", "General",
                "Sgt.", "Sergeant", "Maj.", "Major"
            ],
            
            'positions': [
                # Professional Titles
                "Professor", "Prof.", "Doctor", "Dr.",
                "Director", "Dir.", "Manager", "Mgr.",
                "Secretary", "Sec.", "President", "Pres.",
                "Chairman", "Chair.", "Executive", "Exec.",
                
                # Religious Titles
                "Reverend", "Rev.", "Bishop", "Father",
                "Pastor", "Minister", "Deacon",
                
                # Legal Positions
                "Attorney", "Atty.", "Esquire", "Esq.",
                "Judge", "Justice", "Solicitor", "Barrister"
            ],
            
            'dates': [
                # Months
                "January", "Jan.", "February", "Feb.",
                "March", "Mar.", "April", "Apr.",
                "May", "June", "Jun.", "July", "Jul.",
                "August", "Aug.", "September", "Sept.",
                "October", "Oct.", "November", "Nov.",
                "December", "Dec.",
                
                # Date Formats
                r"\d{1,2}/\d{1,2}/\d{4}",  # e.g., 12/25/1900
                r"\d{1,2}\.\d{1,2}\.\d{4}",  # e.g., 12.25.1900
                r"\d{1,2}-\d{1,2}-\d{4}",  # e.g., 12-25-1900
                
                # Common Date Words
                "Date:", "Dated", "On this day",
                "Anno Domini", "A.D.", "Year of Our Lord"
            ],
            
            'numbers': [
                # Basic Numbering
                "No.", "Number", "Item", "Ref.",
                "Page", "Pg.", "P.", "pp.",
                
                # Roman Numerals
                "I", "II", "III", "IV", "V",
                "VI", "VII", "VIII", "IX", "X",
                
                # Common Number Formats
                r"\d+st",  # 1st
                r"\d+nd",  # 2nd
                r"\d+rd",  # 3rd
                r"\d+th",  # 4th, 5th, etc.
                
                # Reference Numbers
                "Ref. No.", "File No.", "Case No.",
                "Document No.", "ID:", "Ref:"
            ],
            
            'locations': [
                # Address Components
                "Street", "St.", "Avenue", "Ave.",
                "Road", "Rd.", "Lane", "Ln.",
                "Boulevard", "Blvd.", "Square", "Sq.",
                
                # Place Designations
                "City", "Town", "Village", "County",
                "State", "Province", "District", "Region",
                "Territory", "Country", "Kingdom",
                
                # Building/Location Types
                "Building", "Bldg.", "Office", "Room",
                "Suite", "Department", "Dept.", "Floor"
            ],
            
            'signatures': [
                # Signature Lines
                "Signed:", "Signature:", "Sign Here:",
                "Witness:", "Attested:", "Approved by:",
                
                # Signature Qualifiers
                "(Signed)", "(Seal)", "(L.S.)",
                "In witness whereof", "In testimony whereof",
                
                # Authentication
                "Certified by:", "Notarized by:",
                "True Copy", "Certified True Copy",
                "Sworn before me"
            ],
            
            'currency': [
                # Basic Currency
                "Dollars", "$", "Cents", "¢",
                "Pounds", "£", "Shillings", "Pence",
                
                # Currency Patterns
                r"\$\d+", r"£\d+",
                r"\d+ Dollars", r"\d+ Cents",
                r"\d+ Pounds", r"\d+ Shillings"
            ],
            
            'common_abbrev': [
                # Latin Abbreviations
                "etc.", "et al.", "e.g.", "i.e.",
                "viz.", "vs.", "cf.", "N.B.",
                
                # Common Business
                "Inc.", "Ltd.", "Co.", "Corp.",
                "Est.", "Dept.", "Div.",
                
                # Document References
                "ibid.", "op. cit.", "loc. cit.",
                "supra", "infra", "ante", "post",
                
                # Measurements
                "ft.", "in.", "yd.", "mi.",
                "lb.", "oz.", "pt.", "qt.", "gal."
            ]
        })
