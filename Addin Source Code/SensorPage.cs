using System;
using System.ComponentModel;
using CodeStack.SwEx.Common.Attributes;
using CodeStack.SwEx.PMPage.Attributes;
using SolidWorks.Interop.sldworks;
using SolidWorks.Interop.swconst;

// Sensor UI
namespace TestAddin
{
    [Title("Select Main Body To Substract From")]
    internal class SensorPage
    {
        [SelectionBox(swSelectType_e.swSelDATUMPLANES, SolidWorks.Interop.swconst.swSelectType_e.swSelSOLIDBODIES)]
        [Description("Select sketch to scale")]
        [ControlAttribution(swControlBitmapLabelType_e.swBitmapLabel_SelectComponent)] // change icon
        public IBody2 Main { get; set; }


        //[SelectionBox(swSelectType_e.swSelDATUMPLANES, SolidWorks.Interop.swconst.swSelectType_e.swSelCOMPONENTS)]
        //[Description("Select sketch to scale")]
        //[ControlAttribution(swControlBitmapLabelType_e.swBitmapLabel_SelectComponent)] // change icon
        //public IEntity Body { get; set; }
    }
}
