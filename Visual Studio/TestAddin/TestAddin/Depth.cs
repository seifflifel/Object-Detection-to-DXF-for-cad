using CodeStack.SwEx.PMPage.Attributes;
using SolidWorks.Interop.swconst;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SolidWorks.Interop.sldworks;


// CUT UI

namespace TestAddin
{
    internal class Depth
    {

        [Description("Select a face to cut")]
        [SelectionBox(swSelectType_e.swSelDATUMPLANES, SolidWorks.Interop.swconst.swSelectType_e.swSelSKETCHES)]
        public IEntity Reference { get; set; }  //use this for selecting 

        [Description("Enter the Depth of the cut in mm")]
        [ControlAttribution(swControlBitmapLabelType_e.swBitmapLabel_Depth)] //change icon
        [NumberBoxOptions(swNumberboxUnitType_e.swNumberBox_Length, 0, 1000, 0.01, false, 0.1, 0.001, swPropMgrPageNumberBoxStyle_e.swPropMgrPageNumberBoxStyle_Thumbwheel)]
        public Double depth { get; set; }
    }
}
