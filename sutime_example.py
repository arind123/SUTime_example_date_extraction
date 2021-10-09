# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import json
from sutime import SUTime


test_case = '''A key oil refinery for U.S. East Coast consumers is halting operations after escalating environmental scrutiny made it impossible for backers to obtain desperately needed financing.

The owners of the Limetree Bay refinery in the U.S. Virgin Islands announced plans Monday to shut the 200,000-barrel-a-day facility and dismiss more than 250 workers just weeks after a federal crackdown over a series of pollution incidents.

The demise of Limetree Bay is the most dramatic fallout from the Biden administration’s crusade to wean the world’s biggest economy off fossil fuels since the January cancellation of the Keystone XL pipeline project. It’s also emblematic of the challenges facing an industry struggling with shrinking profitability, excess production capacity and rising competition from mega-refineries in Asia.

“There’s no reason we won’t see further closures in the U.S.,” said Robert Campbell, head of oil products research at Energy Aspects Ltd. Refiners will find it harder and harder to raise money for equipment upgrades and pollution-control gear, he noted.

Refinery executives told employees on Monday that 271 of them will lose their jobs effective Sept. 19, according to a company statement that cited “severe financial constraints.”

Limetree Bay has attracted the attention of environmental regulators since its backers that include ArcLight Capital Partners, Freepoint Commodities and EIG Global Energy Partners began efforts to restart the idled refinery in September.

Last month, following a slew of emissions incidents that included contamination of drinking water, the Environmental Protection Agency ordered it to halt operations, reversing a Trump administration approval.

Known formerly as Hovensa, the St. Croix plant was previously owned by Hess Corp. and Venezuela’s state-owned Petroleos de Venezuela SA before it was shuttered in 2012. Once a major supplier of gasoline and diesel to the East Coast markets, the facility was mothballed during a previous downturn in demand and increased international competition.

Roughly 2 million barrels of daily refining capacity may be shut next year to avoid further margin erosion, BloombergNEF analyst Sisi Tang said in a report. The transition away from fossil fuels also dims the long-term outlook for refiners, prompting companies such as Valero Energy Corp. to expand into biofuels.'''
sutime = SUTime(mark_time_ranges=True, include_range=True)
print(json.dumps(sutime.parse(test_case), sort_keys=True, indent=4))


SUTime()