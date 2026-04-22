"""
expert_system.py
────────────────────────────────────────────────────────────────────────────────
CS6053 – Credit Risk Expert System  |  Member B
Rule-based expert system using the `experta` library (forward-chaining).

Three-stage design:
  Stage 1 – Data validation / hard-stop rules  (rules 1-10)
  Stage 2 – Hard affordability & risk rules    (rules 11-20)
  Stage 3 – Manual-review rules                (rules 21-24)

Decisions:
  REJECT – application is declined
  REFER  – application requires manual/enhanced review
  APPROVE – no adverse rules fired; application may proceed

Usage:
  from expert_system import assess_applicant

  result = assess_applicant(
      person_age=22,
      person_income=30000,
      loan_amnt=8000,
      loan_percent_income=0.27,
      loan_grade='B',
      loan_int_rate=11.5,
      cb_person_cred_hist_length=3,
      person_emp_length=2,
      person_home_ownership='RENT',
      loan_intent='EDUCATION',
      cb_person_default_on_file='N',
  )
  print(result)
  # {'decision': 'APPROVE', 'reasons': [...]}
────────────────────────────────────────────────────────────────────────────────
"""

from experta import *


# ── Fact definitions ──────────────────────────────────────────────────────────

class Applicant(Fact):
    """Holds all input fields for a single loan applicant."""
    pass


class Decision(Fact):
    """Holds the outcome and the reason it was triggered."""
    # outcome: 'REJECT' | 'REFER' | 'APPROVE'
    # reason:  human-readable string
    pass


# ── Knowledge Engine ──────────────────────────────────────────────────────────

class CreditRiskEngine(KnowledgeEngine):
    """
    Forward-chaining expert system for credit risk assessment.
    Rules are checked in priority order (salience).
    Once a REJECT or REFER decision is declared the engine continues to
    collect *all* reasons that apply (no early exit) so the output is
    fully explainable.
    """

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1 – Validation rules  (salience 100)
    # ══════════════════════════════════════════════════════════════════════════

    @Rule(Applicant(person_age=MATCH.age),
          TEST(lambda age: age is None),
          salience=100)
    def val_age_missing(self, age):
        self.declare(Decision(outcome='REFER',
                               reason='Rule 1: person_age is missing – cannot assess.'))

    @Rule(Applicant(person_income=MATCH.inc),
          TEST(lambda inc: inc is None or inc <= 0),
          salience=100)
    def val_income_missing(self, inc):
        self.declare(Decision(outcome='REJECT',
                               reason='Rule 2: person_income is missing or ≤ 0.'))

    @Rule(Applicant(loan_amnt=MATCH.amt),
          TEST(lambda amt: amt is None or amt <= 0),
          salience=100)
    def val_loan_amnt_missing(self, amt):
        self.declare(Decision(outcome='REJECT',
                               reason='Rule 3: loan_amnt is missing or ≤ 0.'))

    @Rule(Applicant(loan_percent_income=MATCH.lpi),
          TEST(lambda lpi: lpi is None),
          salience=100)
    def val_lpi_missing(self, lpi):
        self.declare(Decision(outcome='REFER',
                               reason='Rule 4: loan_percent_income is missing.'))

    @Rule(Applicant(loan_grade=MATCH.lg),
          TEST(lambda lg: lg is None),
          salience=100)
    def val_grade_missing(self, lg):
        self.declare(Decision(outcome='REFER',
                               reason='Rule 5: loan_grade is missing.'))

    @Rule(Applicant(loan_int_rate=MATCH.ir),
          TEST(lambda ir: ir is None),
          salience=100)
    def val_int_rate_missing(self, ir):
        self.declare(Decision(outcome='REFER',
                               reason='Rule 6: loan_int_rate is missing.'))

    @Rule(Applicant(cb_person_cred_hist_length=MATCH.chl),
          TEST(lambda chl: chl is None),
          salience=100)
    def val_cred_hist_missing(self, chl):
        self.declare(Decision(outcome='REFER',
                               reason='Rule 7: cb_person_cred_hist_length is missing.'))

    @Rule(Applicant(person_emp_length=MATCH.emp),
          TEST(lambda emp: emp is None),
          salience=100)
    def val_emp_missing(self, emp):
        self.declare(Decision(outcome='REFER',
                               reason='Rule 8: person_emp_length is missing.'))

    @Rule(Applicant(person_age=MATCH.age, person_emp_length=MATCH.emp),
          TEST(lambda age, emp: age is not None and emp is not None and emp > age - 16),
          salience=100)
    def val_emp_exceeds_age(self, age, emp):
        self.declare(Decision(outcome='REFER',
                               reason='Rule 9: person_emp_length exceeds person_age − 16 '
                                      '(data quality issue).'))

    @Rule(Applicant(person_age=MATCH.age),
          TEST(lambda age: age is not None and age < 18),
          salience=100)
    def val_underage(self, age):
        self.declare(Decision(outcome='REJECT',
                               reason='Rule 10: Applicant is under 18 – ineligible for credit.'))

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2 – Hard affordability & risk rules  (salience 80)
    # ══════════════════════════════════════════════════════════════════════════

    @Rule(Applicant(loan_percent_income=MATCH.lpi),
          TEST(lambda lpi: lpi is not None and lpi >= 0.50),
          salience=80)
    def risk_lpi_very_high(self, lpi):
        self.declare(Decision(outcome='REJECT',
                               reason='Rule 11: loan_percent_income ≥ 50% – repayment '
                                      'burden is likely unsustainable.'))

    @Rule(Applicant(loan_percent_income=MATCH.lpi, person_income=MATCH.inc),
          TEST(lambda lpi, inc: lpi is not None and inc is not None
                                and lpi >= 0.40 and inc < 25000),
          salience=80)
    def risk_lpi_high_low_income(self, lpi, inc):
        self.declare(Decision(outcome='REJECT',
                               reason='Rule 12: loan_percent_income ≥ 40% combined with '
                                      'income < £25,000.'))

    @Rule(Applicant(cb_person_default_on_file=MATCH.def_,
                    loan_grade=MATCH.lg),
          TEST(lambda def_, lg: def_ == 'Y' and lg in ('D', 'E', 'F', 'G')),
          salience=80)
    def risk_default_poor_grade(self, def_, lg):
        self.declare(Decision(outcome='REJECT',
                               reason=f'Rule 13: Prior default on file with loan grade '
                                      f'{lg} (D–G band).'))

    @Rule(Applicant(cb_person_default_on_file=MATCH.def_,
                    loan_percent_income=MATCH.lpi),
          TEST(lambda def_, lpi: def_ == 'Y' and lpi is not None and lpi > 0.35),
          salience=80)
    def risk_default_high_lpi(self, def_, lpi):
        self.declare(Decision(outcome='REJECT',
                               reason='Rule 14: Prior default on file and '
                                      'loan_percent_income > 35%.'))

    @Rule(Applicant(cb_person_cred_hist_length=MATCH.chl,
                    loan_amnt=MATCH.amt),
          TEST(lambda chl, amt: chl is not None and amt is not None
                                and chl < 1 and amt > 10000),
          salience=80)
    def risk_no_history_large_loan(self, chl, amt):
        self.declare(Decision(outcome='REJECT',
                               reason='Rule 15: Credit history < 1 year and '
                                      'loan amount > £10,000.'))

    @Rule(Applicant(person_emp_length=MATCH.emp,
                    loan_percent_income=MATCH.lpi),
          TEST(lambda emp, lpi: emp is not None and lpi is not None
                                and emp == 0 and lpi > 0.35),
          salience=80)
    def risk_unemployed_high_lpi(self, emp, lpi):
        self.declare(Decision(outcome='REJECT',
                               reason='Rule 16: Zero employment length and '
                                      'loan_percent_income > 35%.'))

    @Rule(Applicant(loan_int_rate=MATCH.ir),
          TEST(lambda ir: ir is not None and ir > 20),
          salience=80)
    def risk_very_high_rate(self, ir):
        self.declare(Decision(outcome='REFER',
                               reason='Rule 17: loan_int_rate > 20% – high-cost credit '
                                      'triggers stricter scrutiny under FCA Consumer Duty.'))

    @Rule(Applicant(person_home_ownership=MATCH.own,
                    loan_percent_income=MATCH.lpi),
          TEST(lambda own, lpi: own == 'OTHER' and lpi is not None and lpi > 0.35),
          salience=80)
    def risk_other_ownership_high_lpi(self, own, lpi):
        self.declare(Decision(outcome='REFER',
                               reason='Rule 18: Home ownership = OTHER and '
                                      'loan_percent_income > 35%.'))

    @Rule(Applicant(loan_intent=MATCH.intent,
                    loan_percent_income=MATCH.lpi),
          TEST(lambda intent, lpi: intent == 'DEBTCONSOLIDATION'
                                   and lpi is not None and lpi > 0.35),
          salience=80)
    def risk_debt_consolidation_high_lpi(self, intent, lpi):
        self.declare(Decision(outcome='REFER',
                               reason='Rule 19: Loan intent = DEBTCONSOLIDATION and '
                                      'loan_percent_income > 35% – may indicate '
                                      'existing financial stress.'))

    @Rule(Applicant(loan_intent=MATCH.intent,
                    loan_percent_income=MATCH.lpi),
          TEST(lambda intent, lpi: intent == 'PERSONAL'
                                   and lpi is not None and lpi > 0.40),
          salience=80)
    def risk_personal_very_high_lpi(self, intent, lpi):
        self.declare(Decision(outcome='REJECT',
                               reason='Rule 20: Loan intent = PERSONAL and '
                                      'loan_percent_income > 40%.'))

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 3 – Manual-review rules  (salience 60)
    # ══════════════════════════════════════════════════════════════════════════

    @Rule(Applicant(person_age=MATCH.age),
          TEST(lambda age: age is not None and 18 <= age <= 20),
          salience=60)
    def review_young_applicant(self, age):
        self.declare(Decision(outcome='REFER',
                               reason=f'Rule 21: Applicant age {age} (18–20) – refer '
                                      'unless score is strong and all checks pass.'))

    @Rule(Applicant(person_age=MATCH.age),
          TEST(lambda age: age is not None and age > 70),
          salience=60)
    def review_elderly_applicant(self, age):
        self.declare(Decision(outcome='REFER',
                               reason=f'Rule 22: Applicant age {age} (>70) – enhanced '
                                      'review for income sustainability and vulnerability.'))

    @Rule(Applicant(person_income=MATCH.inc, person_age=MATCH.age),
          TEST(lambda inc, age: inc is not None and age is not None
                                and inc > 250000 and age < 25),
          salience=60)
    def review_high_income_young(self, inc, age):
        self.declare(Decision(outcome='REFER',
                               reason=f'Rule 23: Income > £250,000 with age {age} (<25) – '
                                      'refer for income verification and KYC/fraud review.'))

    @Rule(Applicant(aml_concern=MATCH.aml),
          TEST(lambda aml: aml is True),
          salience=60)
    def review_aml_concern(self, aml):
        self.declare(Decision(outcome='REFER',
                               reason='Rule 24: AML / identity / source-of-funds concern '
                                      'flagged – customer due diligence required.'))

    # ══════════════════════════════════════════════════════════════════════════
    # Default rule – APPROVE if nothing else fired
    # ══════════════════════════════════════════════════════════════════════════

    @Rule(NOT(Decision(outcome=W())),
          salience=-1)
    def default_approve(self):
        self.declare(Decision(outcome='APPROVE',
                               reason='No adverse rules triggered – application may proceed.'))


# ── Public interface ──────────────────────────────────────────────────────────

def assess_applicant(
    person_age=None,
    person_income=None,
    loan_amnt=None,
    loan_percent_income=None,
    loan_grade=None,
    loan_int_rate=None,
    cb_person_cred_hist_length=None,
    person_emp_length=None,
    person_home_ownership=None,
    loan_intent=None,
    cb_person_default_on_file=None,
    aml_concern=False,
) -> dict:
    """
    Run the expert system for a single applicant.

    Parameters
    ----------
    All feature values as keyword arguments.  Pass None for any missing field.
    loan_grade          : str  – e.g. 'A', 'B', 'C', 'D', 'E', 'F', 'G'
    cb_person_default_on_file : str – 'Y' or 'N'
    person_home_ownership     : str – 'RENT', 'OWN', 'MORTGAGE', 'OTHER'
    loan_intent               : str – 'PERSONAL', 'EDUCATION', 'MEDICAL',
                                       'VENTURE', 'HOMEIMPROVEMENT',
                                       'DEBTCONSOLIDATION'
    aml_concern               : bool – True if external AML flag raised

    Returns
    -------
    dict with keys:
        'decision' : 'APPROVE' | 'REFER' | 'REJECT'
        'reasons'  : list of str  (all rules that fired)
    """
    engine = CreditRiskEngine()
    engine.reset()
    engine.declare(Applicant(
        person_age=person_age,
        person_income=person_income,
        loan_amnt=loan_amnt,
        loan_percent_income=loan_percent_income,
        loan_grade=loan_grade,
        loan_int_rate=loan_int_rate,
        cb_person_cred_hist_length=cb_person_cred_hist_length,
        person_emp_length=person_emp_length,
        person_home_ownership=person_home_ownership,
        loan_intent=loan_intent,
        cb_person_default_on_file=cb_person_default_on_file,
        aml_concern=aml_concern,
    ))
    engine.run()

    decisions = [f for f in engine.facts.values() if isinstance(f, Decision)]

    # Priority: REJECT > REFER > APPROVE
    priority = {'REJECT': 2, 'REFER': 1, 'APPROVE': 0}
    final = max(decisions, key=lambda d: priority.get(d['outcome'], 0))

    reasons = [d['reason'] for d in decisions]

    return {
        'decision': final['outcome'],
        'reasons': reasons,
    }


# ── Quick demo ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    test_cases = [
        {
            'label': 'Clear REJECT – high LPI + low income',
            'kwargs': dict(person_age=35, person_income=20000, loan_amnt=9000,
                           loan_percent_income=0.45, loan_grade='C',
                           loan_int_rate=13.0, cb_person_cred_hist_length=4,
                           person_emp_length=3, person_home_ownership='RENT',
                           loan_intent='PERSONAL', cb_person_default_on_file='N'),
        },
        {
            'label': 'Clear APPROVE – strong profile',
            'kwargs': dict(person_age=40, person_income=60000, loan_amnt=10000,
                           loan_percent_income=0.17, loan_grade='B',
                           loan_int_rate=9.5, cb_person_cred_hist_length=8,
                           person_emp_length=10, person_home_ownership='MORTGAGE',
                           loan_intent='HOMEIMPROVEMENT', cb_person_default_on_file='N'),
        },
        {
            'label': 'REFER – young applicant (age 19)',
            'kwargs': dict(person_age=19, person_income=22000, loan_amnt=3000,
                           loan_percent_income=0.14, loan_grade='A',
                           loan_int_rate=7.5, cb_person_cred_hist_length=1,
                           person_emp_length=1, person_home_ownership='RENT',
                           loan_intent='EDUCATION', cb_person_default_on_file='N'),
        },
        {
            'label': 'REJECT – prior default + grade E',
            'kwargs': dict(person_age=45, person_income=35000, loan_amnt=12000,
                           loan_percent_income=0.34, loan_grade='E',
                           loan_int_rate=18.0, cb_person_cred_hist_length=5,
                           person_emp_length=6, person_home_ownership='RENT',
                           loan_intent='DEBTCONSOLIDATION', cb_person_default_on_file='Y'),
        },
        {
            'label': 'REJECT – underage applicant',
            'kwargs': dict(person_age=16, person_income=10000, loan_amnt=2000,
                           loan_percent_income=0.20, loan_grade='A',
                           loan_int_rate=8.0, cb_person_cred_hist_length=0,
                           person_emp_length=0, person_home_ownership='RENT',
                           loan_intent='PERSONAL', cb_person_default_on_file='N'),
        },
    ]

    for case in test_cases:
        result = assess_applicant(**case['kwargs'])
        print(f"\n{'─'*60}")
        print(f"TEST:     {case['label']}")
        print(f"DECISION: {result['decision']}")
        print("REASONS:")
        for r in result['reasons']:
            print(f"  • {r}")
    print(f"\n{'─'*60}")